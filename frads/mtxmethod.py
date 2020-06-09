
"""
T.Wang

"""

import os
import copy
import tempfile as tf
from frads import mfacade, radgeom, room
from frads import radutil, radmtx, makesky
import multiprocessing as mp
import logging

logger = logging.getLogger('frads.mtxmethod')

cfg_template = {
    'SimulationControl':{
        'vmx_opt': 'kf -ab 1 -ad 512', 'fmx_opt': None, 'dmx_opt': 'r4 -ab 1 -ad 128 -c 2000',
        'dsmx_opt': 'r4 -ab 3 -ad 262144 -lw 1e-9', 'cdsmx_opt': 'r6 -ab 1', 'ray_count': 1,
        'pixel_jitter': .7, 'separate_direct': False, 'nprocess': 1,
    }, 'FileStructure':{
        'base': '', 'matrices': 'Matrices', 'results': 'Results',
        'objects': 'Objects', 'resources': 'Resources',
    }, 'Site':{
        'wea': None, 'latitude': None, 'longitude': None, 'zipcode': None,
        'daylight_hours_only': False, 'start_hour': None, 'end_hour': None,
    }, 'Dimensions':{
        'depth': None, 'width': None, 'height': None, 'window1': None,
        'facade_thickness': None, 'orientation': None,
    }, 'Model':{
        'material': None, 'windows': None, 'scene': None,
        'ncp_shade': None, 'BSDF': None, 'sunBSDF': None,
    }, 'Raysenders':{
        'view1': None, 'grid_surface': None, 'distance': None,
        'spacing': None, 'opposite': True,
    }}


class MTXmethod(object):
    def __init__(self, config):
        self.logger = logging.getLogger("frads.mtxmethod.MTXmethod")
        self.nproc = int(config.simctrl['nprocess'])
        self.vmx_opt = config.simctrl['vmx_opt'][3:] + f' -n {self.nproc}'
        self.vmx_basis = config.simctrl['vmx_opt'][:2]
        self.dmx_opt = config.simctrl['dmx_opt'][3:] + f' -n {self.nproc}'
        self.dmx_basis = config.simctrl['dmx_opt'][:2]
        self.dsmx_opt = config.simctrl['dsmx_opt'][3:] + f' -n {self.nproc}'
        self.dsmx_basis = config.simctrl['dsmx_opt'][:2]
        self.cdsmx_opt = config.simctrl['cdsmx_opt'][3:] + f' -n {self.nproc}'
        self.cdsmx_basis = config.simctrl['cdsmx_opt'][:2]
        self.ray_cnt = int(config.simctrl['ray_count'])
        # get directories
        self.mtxdir = config.mtxdir
        self.resdir = os.path.join(config.filestrct['base'], config.filestrct['results'])
        get paths
        self.materialpath = config.materialpath
        self.envpath = config.scenepath
        self.envpath.insert(0, self.materialpath)
        self.windowpath = config.windowpath
        self.wea_path = config.wea_path
        # get primitives and others
        self.scene_prims = config.scene_prims
        self.window_prims = config.window_prims
        self.sensor_pts = config.sensor_pts
        self.bsdf = config.bsdf
        self.dts = config.dts
        self.td = tf.mkdtemp()
        if self.sensor_pts is not None:
            self.sndr_pts = radmtx.Sender.as_pts(
                pts_list=self.sensor_pts, ray_cnt=self.ray_cnt, tmpdir=self.td)
        self.sndr_vus = {}
        if len(config.viewdicts) > 0:
            for view in config.viewdicts:
                self.sndr_vus[view] = radmtx.Sender.as_view(
                    vu_dict=config.viewdicts[view], ray_cnt=self.ray_cnt, tmpdir=self.td,
                    xres=config.viewdicts[view]['x'], yres=config.viewdicts[view]['y'])
        self.rcvr_sky = radmtx.Receiver.as_sky(basis=self.dmx_basis, tmpdir=self.td)

    def prep_2phase(self):
        env = self.envpath + self.windowpath
        if self.sensor_pts is not None:
            self.pdsmx = os.path.join(self.mtxdir, 'pdsmx.mtx')
            radmtx.rfluxmtx(sender=sndr_pts, receiver=self.rcvr_sky,
                            env=env, out=self.pdsmx, opt=self.dsmx_opt)
        if len(self.sndr_vus) > 0:
            self.vdsmxs = {}
            for vu in self.sndr_vus:
                self.vdsmxs[vu] = os.path.join(self.mtxdir, f"vdsmx_{vu}")
                radmtx.rfluxmtx(sender=self.sndr_vus[vu], receiver=self.rcvr_sky,
                                env=env, opt=self.dsmx_opt, out=self.vdsmxs[vu])

    def calc_2phase(self):
        self.gen_smx(self.dmx_basis)
        if self.sensor_pts is not None:
            res = self.mtxmult('dctimestep', self.pdsmx, self.smxpath).splitlines()
            respath = os.path.join(self.resdir, 'pdsmx.txt')
            with open(respath, 'w') as wtr:
                for idx in range(len(res)):
                    wtr.write(self.dts[idx] + '\t')
                    wtr.write(res[idx] + os.linesep)
        if len(self.sndr_vus) > 0:
            opath = os.path.join(self.resdir, 'view2ph')
            radutil.sprun(self.imgmult(self.vdsmx, self.smxpath, odir=opath))
            ofiles = [os.path.join(opath, f) for f in sorted(os.listdir(opath)) if
                      f.endswith('.hdr')]
            [os.rename(ofiles[idx], os.path.join(opath, self.dts[idx]+'.hdr'))
             for idx in range(len(ofiles))]

    def prep_3phase(self):
        self.pvmxs = {}
        self.vvmxs = {}
        self.dmxs = {}
        prcvr_wndws = radmtx.Receiver(
            path=None, receiver='', basis=self.vmx_basis, modifier=None)
        if len(self.sndr_vus) > 0:
            vrcvr_wndws = {}
            for vu in self.sndr_vus:
                vrcvr_wndws[vu] = radmtx.Receiver(
                    path=None, receiver='', basis=self.vmx_basis, modifier=None)
        for wname in self.window_prims:
            wndw_prim = self.window_prims[wname]
            self.logger.info(f"Generating daylight matrix for {wname}")
            self.dmxs[wname] = os.path.join(self.mtxdir, f'dmx_{wname}.mtx')
            sndr_wndw = radmtx.Sender.as_surface(tmpdir=self.td,
                prim_list=wndw_prim, basis=self.vmx_basis, offset=None)
            radmtx.rfluxmtx(sender=sndr_wndw, receiver=self.rcvr_sky,
                       env=self.envpath, out=self.dmxs[wname], opt=self.dmx_opt)
            if self.sensor_pts is not None:
                self.pvmxs[wname] = os.path.join(self.mtxdir, f'pvmx_{wname}.mtx')
                prcvr_wndws += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                    offset=None, left=None, source='glow', out=self.pvmxs[wname])
            if len(self.sndr_vus) > 0:
                for vu in self.sndr_vus:
                    self.vvmxs[vu+wname] = os.path.join(
                        self.mtxdir, f'vvmx_{vu}_{wname}', '%04d.hdr')
                    radutil.mkdir_p(os.path.dirname(self.vvmxs[vu+wname]))
                    vrcvr_wndws[vu] += radmtx.Receiver.as_surface(
                        prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                        offset=None, left=None, source='glow', out=self.vvmxs[vu+wname])
        if self.sensor_pts is not None:
            self.logger.info("Generating view matrix for sensor point")
            radmtx.rfluxmtx(sender=self.sndr_pts, receiver=prcvr_wndws,
                            env=self.envpath, opt=self.vmx_opt, out=None)
        if len(self.sndr_vus) > 0:
            self.logger.info(f"Generating view matrix for {vu}")
            for vu in self.sndr_vus:
                radmtx.rfluxmtx(sender=self.sndr_vus[vu], receiver=vrcvr_wndws[vu],
                                env=self.envpath, opt=self.vmx_opt, out=None)

    def calc_3phase(self):
        self.gen_smx(self.dmx_basis)
        if self.sensor_pts is not None:
            self.logger.info("Computing for point grid results")
            presl = []
            for wname in self.window_prims:
                _res = radutil.spcheckout(
                    self.mtxmult(self.pvmxs[wname], self.bsdf[wname],
                                 self.dmxs[wname], self.smxpath))
                presl.append([map(float, l.strip().split('\t')) for l in _res.splitlines()])
            res = [[sum(tup) for tup in zip(*line)]for line in zip(*presl)]
            respath = os.path.join(self.resdir, 'points3ph.txt')
            with open(respath, 'w') as wtr:
                for idx in range(len(res)):
                    wtr.write(self.dts[idx] + ',')
                    wtr.write(','.join(map(str, res[idx])) + os.linesep)
        if len(self.sndr_vus) > 0:
            for vu in self.sndr_vus:
                self.logger.info(f"Computing for rendering results for {vu}")
                vresl = []
                for wname in self.window_prims:
                    _vrespath = tf.mkdtemp(dir=self.td)
                    self.imgmult(self.vvmxs[vu+wname], self.bsdf[wname], self.dmxs[wname],
                              self.smxpath, odir=_vrespath)
                    vresl.append(_vrespath)
                opath = os.path.join(self.resdir, f'{vu}_3ph')
                if len(self.window_prims)>1:
                    [vresl.insert(i*2-1, '+') for i in range(1, len(vresl)+1)]
                    radutil.pcombop(vresl, opath)
                else:
                    os.rename(vresl[0], opath)
                ofiles = [os.path.join(opath, f) for f in sorted(os.listdir(opath)) if
                          f.endswith('.hdr')]
                [os.rename(ofiles[idx], os.path.join(opath, self.dts[idx]+'.hdr'))
                 for idx in range(len(ofiles))]

    def prep_4phase(self):
        self.pvmxs = {}
        self.vvmxs = {}
        self.dmxs = {}
        self.fmxs = {}
        prcvr_wndws = radmtx.Receiver(
            path=None, receiver='', basis=self.vmx_basis, modifier=None)
        if len(self.sndr_vus) > 0:
            vrcvr_wndws = {}
            for vu in self.sndr_vus:
                vrcvr_wndws[vu] = radmtx.Receiver(
                    path=None, receiver='', basis=self.vmx_basis, modifier=None)
        port_prims = mfacade.genport(
            wpolys=wndw_prims, npolys=ncp_prims, depth=None, scale=None)
        mfacade.Genfmtx(win_polygons=wndw_polygon, port_prim=port_prims,
                        out=kwargs['o'], env=kwargs['env'], sbasis=kwargs['ss'],
                        rbasis=kwargs['rs'], opt=kwargs['opt'], refl=False,
                        forw=False, wrap=False)
        for wname in self.window_prims:
            wndw_prim = self.window_prims[wname]
            self.logger.info(f"Generating daylight matrix for {wname}")
            self.dmxs[wname] = os.path.join(self.mtxdir, f'dmx_{wname}.mtx')
            sndr_wndw = radmtx.Sender.as_surface(tmpdir=self.td,
                prim_list=wndw_prim, basis=self.vmx_basis, offset=None)
            sndr_port = radmtx.Sender.as_surface(tmpdir=self.td,
                prim_list=port_prims, basis=self.fmx_basis, offset=None)
            radmtx.rfluxmtx(sender=sndr_port, receiver=self.rcvr_sky,
                       env=self.envpath, out=self.dmxs[wname], opt=self.dmx_opt)
            if self.sensor_pts is not None:
                self.pvmxs[wname] = os.path.join(self.mtxdir, f'pvmx_{wname}.mtx')
                prcvr_wndws += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                    offset=None, left=None, source='glow', out=self.pvmxs[wname])
            if len(self.sndr_vus) > 0:
                for vu in self.sndr_vus:
                    self.vvmxs[vu+wname] = os.path.join(
                        self.mtxdir, f'vvmx_{vu}_{wname}', '%04d.hdr')
                    radutil.mkdir_p(os.path.dirname(self.vvmxs[vu+wname]))
                    vrcvr_wndws[vu] += radmtx.Receiver.as_surface(
                        prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                        offset=None, left=None, source='glow', out=self.vvmxs[vu+wname])
        if self.sensor_pts is not None:
            self.logger.info("Generating view matrix for sensor point")
            radmtx.rfluxmtx(sender=self.sndr_pts, receiver=prcvr_wndws,
                            env=self.envpath, opt=self.vmx_opt, out=None)
        if len(self.sndr_vus) > 0:
            self.logger.info(f"Generating view matrix for {vu}")
            for vu in self.sndr_vus:
                radmtx.rfluxmtx(sender=self.sndr_vus[vu], receiver=vrcvr_wndws[vu],
                                env=self.envpath, opt=self.vmx_opt, out=None)

    def blacken_env(self):
        blackened = copy.deepcopy(self.scene_prims)
        for prim in blackened:
            prim['modifier'] = 'black'
        _, self.blackenvpath = tf.mkstemp(prefix='blackenv')
        with open(self.blackenvpath, 'w') as wtr:
            for prim in blackened:
                wtr.write(radutil.put_primitive(prim))
        if len(self.sndr_vus) > 0:
            _, blacken_wndw = tf.mkstemp(dir=self.td)
            _, glowing_wndw = tf.mkstemp(dir=self.td)
            blackwindow = copy.deepcopy(self.window_prims)
            glowwindow = copy.deepcopy(self.window_prims)
            for wname in blackwindow:
                for prim in blackwindow[wname]:
                    prim['modifier'] = 'black'
                for prim in glowwindow[wname]:
                    prim['modifier'] = 'glowing'
            with open(blacken_wndw, 'w') as wtr:
                for wndw in blackwindow:
                    [wtr.write(radutil.put_primitive(prim)) for prim in blackwindow[wndw]]
            with open(glowing_wndw, 'w') as wtr:
                for wndw in glowwindow:
                    [wtr.write(radutil.put_primitive(prim)) for prim in glowwindow[wndw]]
            _, self.map1_oct = tf.mkstemp(dir=self.td)
            _, self.map2_oct = tf.mkstemp(dir=self.td)
            radutil.sprun(f"oconv {' '.join(self.envpath)} {glowing_wndw} > {self.map1_oct}")
            radutil.sprun(f"oconv {' '.join(self.envpath)} {blacken_wndw} > {self.map2_oct}")


    def prep_5phase(self):
        self.prep_3phase()
        self.blacken_env()
        self.gen_smx(self.dmx_basis, direct=True)
        self.gen_smx(self.cdsmx_basis, onesun=True, direct=True)
        rcvr_sun = radmtx.Receiver.as_sun(
            basis='r6', tmpdir=self.td, smx_path=self.smxpath,
            window_paths=self.windowpath)
        prcvr_wndws = radmtx.Receiver(
            path=None, receiver='', basis=self.vmx_basis, modifier=None)
        if len(self.sndr_vus) > 0:
            vdrcvr_wndws = {}
            for vu in self.sndr_vus:
                vdrcvr_wndws[vu] = radmtx.Receiver(
                    path=None, receiver='', basis=self.vmx_basis, modifier=None)
        cdsenv = [self.materialpath, self.blackenvpath] + self.windowpath
        for wname in self.window_prims:
            wndw_prim = self.window_prims[wname]
            sndr_wndw = radmtx.Sender.as_surface(tmpdir=self.td,
                prim_list=wndw_prim, basis=self.vmx_basis, offset=None)
            self.dmxs[wname+'_d'] = os.path.join(self.mtxdir, f'dmx_{wname}_d.mtx')
            self.logger.info(f"Generating direct daylight matrix for {wname}")
            radmtx.rfluxmtx(sender=sndr_wndw, receiver=self.rcvr_sky,
                            env=[self.materialpath, self.blackenvpath],
                            out=self.dmxs[wname+'_d'], opt=self.dmx_opt+' -ab 0')
            if self.sensor_pts is not None:
                self.pvmxs[wname+'_d'] = os.path.join(self.mtxdir, f'pvmx_{wname}_d.mtx')
                prcvr_wndws += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                    offset=None, left=None, source='glow', out=self.pvmxs[wname+'_d'])
            if len(self.sndr_vus) > 0:
                for vu in self.sndr_vus:
                    self.vvmxs[vu+wname+'_d'] = os.path.join(
                        self.mtxdir, f'vvmx_{vu}_{wname}_d', '%04d.hdr')
                    radutil.mkdir_p(os.path.dirname(self.vvmxs[vu+wname+'_d']))
                    vdrcvr_wndws[vu] += radmtx.Receiver.as_surface(
                        prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                        offset=None, left=None, source='glow',
                        out=self.vvmxs[vu+wname+'_d'])
        if self.sensor_pts is not None:
            self.logger.info(f"Generating direct view matrix for sensor grid")
            radmtx.rfluxmtx(sender=self.sndr_pts, receiver=prcvr_wndws,
                            env=[self.materialpath, self.blackenvpath],
                            out=None, opt=self.vmx_opt+' -ab 1')
            self.logger.info(f"Generating direct sun matrix for sensor_grid")
            self.pcdsmx = os.path.join(self.mtxdir, 'pcdsmx.mtx')
            radmtx.rcontrib(sender=self.sndr_pts, receiver=rcvr_sun,
                            env=cdsenv, out=self.pcdsmx, opt=self.cdsmx_opt)
        if len(self.sndr_vus) > 0:
            self.vcdfmx = {}
            self.vcdrmx = {}
            self.map1_paths = {}
            self.map2_paths = {}
            for vu in self.sndr_vus:
                self.map1_paths[vu] = os.path.join(self.mtxdir, f'map1_{vu}.hdr')
                self.map2_paths[vu] = os.path.join(self.mtxdir, f'map2_{vu}.hdr')
                vdict = self.sndr_vus[vu].sender
                vdict.pop('c', None)
                vdict.pop('pj', None)
                vu_str = radutil.opt2str(vdict)
                cmd = f"rpict {vu_str} -ps 1 -ab 0 -av .31831 .31831 .31831 "
                cmd += f"{self.map1_oct} > {self.map1_paths[vu]}"
                radutil.sprun(cmd)
                cmd = f"rpict {vu_str} -ps 1 -ab 0 -av .31831 .31831 .31831 "
                cmd += f"{self.map2_oct} > {self.map2_paths[vu]}"
                radutil.sprun(cmd)
                self.logger.info(f"Generating direct view matrix for {vu}")
                radmtx.rfluxmtx(sender=self.sndr_vus[vu], receiver=vdrcvr_wndws[vu],
                                env=[self.materialpath, self.blackenvpath],
                                opt=self.vmx_opt+' -ab 1 -i', out=None)
                self.vcdfmx[vu] = os.path.join(self.mtxdir, f'vcdrmx_{vu}')
                self.vcdrmx[vu] = os.path.join(self.mtxdir, f'vcdfmx_{vu}')
                self.logger.info(f"Generating direct sun r matrix for {vu}")
                radmtx.rcontrib(sender=self.sndr_vus[vu], receiver=rcvr_sun,
                                env=cdsenv, out=self.vcdrmx[vu], opt=self.cdsmx_opt+' -i')
                self.logger.info(f"Generating direct sun f matrix for {vu}")
                radmtx.rcontrib(sender=self.sndr_vus[vu], receiver=rcvr_sun,
                                env=cdsenv, out=self.vcdfmx[vu], opt=self.cdsmx_opt)

    def calc_5phase(self):
        if self.sensor_pts is not None:
            presl = []
            pdresl = []
            mult_cds = radutil.spcheckout(self.mtxmult(self.pcdsmx, self.smxd6path))
            prescd = [list(map(float, l.strip().split('\t')))
                     for l in mult_cds.splitlines()]
            for wname in self.window_prims:
                _res = radutil.spcheckout(self.mtxmult(self.pvmxs[wname], self.bsdf[wname],
                                   self.dmxs[wname], self.smxpath))
                _resd = radutil.spcheckout(self.mtxmult(self.pvmxs[wname+'_d'], self.bsdf[wname],
                                   self.dmxs[wname+'_d'], self.smxdpath))
                presl.append([map(float, l.strip().split('\t')) for l in _res.splitlines()])
                pdresl.append([map(float, l.strip().split('\t')) for l in _resd.splitlines()])
            pres3 = [[sum(tup) for tup in zip(*line)]for line in zip(*presl)]
            pres3d = [[sum(tup) for tup in zip(*line)]for line in zip(*pdresl)]
            res = [[x-y+z for x,y,z in zip(a,b,c)] for a,b,c in zip(pres3, pres3d, prescd)]
            respath = os.path.join(self.resdir, 'points5ph.txt')
            with open(respath, 'w') as wtr:
                for idx in range(len(res)):
                    wtr.write(self.dts[idx] + ',')
                    wtr.write(','.join(map(str, res[idx])) + os.linesep)
        if len(self.sndr_vus) > 0:
            for vu in self.sndr_vus:
                self.logger.info(f"Computing for rendering results for {vu}")
                vresl = []
                vdresl = []
                vrescdr = tf.mkdtemp(dir=self.td)
                vrescdf = tf.mkdtemp(dir=self.td)
                vrescd = tf.mkdtemp(dir=self.td)
                cmds = []
                cmds.append(self.imgmult(os.path.join(self.vcdrmx[vu], '%04d.hdr'),
                                         self.smxd6path, odir=vrescdr))
                cmds.append(self.imgmult(os.path.join(self.vcdfmx[vu], '%04d.hdr'),
                                         self.smxd6path, odir=vrescdf))
                for wname in self.window_prims:
                    _vrespath = tf.mkdtemp(dir=self.td)
                    _vdrespath = tf.mkdtemp(dir=self.td)
                    cmds.append(self.imgmult(self.vvmxs[vu+wname], self.bsdf[wname],
                                 self.dmxs[wname], self.smxpath, odir=_vrespath))
                    cmds.append(self.imgmult(self.vvmxs[vu+wname+'_d'], self.bsdf[wname],
                                 self.dmxs[wname+'_d'], self.smxdpath, odir=_vdrespath))
                    vresl.append(_vrespath)
                    vdresl.append(_vdrespath)
                process = mp.Pool(self.nproc)
                process.map(radutil.sprun, cmds)
                res3 = tf.mkdtemp(dir=self.td)
                res3di = tf.mkdtemp(dir=self.td)
                res3d = tf.mkdtemp(dir=self.td)
                if len(self.window_prims)>1:
                    [vresl.insert(i*2-1, '+') for i in range(1, len(vresl)+1)]
                    [vdresl.insert(i*2-1, '+') for i in range(1, len(vdresl)+1)]
                    radutil.pcombop(vresl, res3)
                    radutil.pcombop(vdresl, res3di)
                else:
                    os.rename(vresl[0], res3)
                    os.rename(vdresl[0], res3di)
                radutil.pcombop([res3di, '*', self.map1_paths[vu]], res3d)
                radutil.pcombop([vrescdr, '*', self.map2_paths[vu], '+', vrescdf], vrescd)
                opath = os.path.join(self.resdir, f"{vu}_5ph")
                if os.path.isdir(opath): shutil.rmtree(opath)
                radutil.pcombop([res3, '-', res3d, '+', vrescd], opath)
                ofiles = [os.path.join(opath, f) for f in sorted(os.listdir(opath)) if
                          f.endswith('.hdr')]
                [os.rename(ofiles[idx], os.path.join(opath, self.dts[idx]+'.hdr'))
                 for idx in range(len(ofiles))]
                self.logger.info(f"Done computing for {vu}")

    def gen_smx(self, mf, onesun=False, direct=False):
        sun_only = ' -d' if direct else ''
        _five = ' -5 .533' if onesun else ''
        oname = radutil.basename(self.wea_path)
        cmd = f"gendaymtx -of -m {mf[-1]}{sun_only}{_five}".split()
        cmd.append(self.wea_path)
        res = radutil.spcheckout(cmd)
        if direct:
            if onesun:
                self.smxd6path = os.path.join(self.mtxdir, oname+'_d6.smx')
                opath = self.smxd6path
            else:
                self.smxdpath = os.path.join(self.mtxdir, oname+'_d.smx')
                opath = self.smxdpath
        else:
            self.smxpath = os.path.join(self.mtxdir, oname+'.smx')
            opath = self.smxpath
        with open(opath, 'wb') as wtr:
            wtr.write(res)

    def mtxmult(self, *mtx):
        cmd = f"dctimestep {' '.join(mtx)} "
        cmd += f'| rmtxop -fd -c 47.4 119.9 11.6 - | rmtxop -fa -t - | getinfo -'
        return cmd

    def imgmult(self, *mtx, odir):
        cmd = f"dctimestep -oc -o {os.path.join(odir, '%04d.hdr')} {' '.join(mtx)}"
        return cmd

    def get_avgskv(self):
        radutil.sprun(f"gendaymtx -m {self.mf_sky} -A {self.wea} > {avgskyv}", shell=True)

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
    def __init__(self, config):
        self.logger = logging.getLogger('frads.mtxmethod.Prepare')
        self.site = config['Site']
        self.filestrct = config['FileStructure']
        self.simctrl = config['SimulationControl']
        self.model = config['Model']
        self.dimensions = config['Dimensions']
        self.raysenders = config['Raysenders']
        self.get_paths()
        if None in self.dimensions.values():
            self.assemble()
        else:
            self.logger.info("Generating a room based on defined dimensions")
            self.make_room()
        self.get_wea()

    def get_paths(self):
        """Where are ?"""
        objdir = os.path.join(self.filestrct['base'], self.filestrct['objects'])
        self.mtxdir = os.path.join(self.filestrct['base'], self.filestrct['matrices'])
        self.materialpath = os.path.join(objdir, self.model['material'])
        self.scenepath = [os.path.join(objdir, obj)
                          for obj in self.model['scene'].split()]
        self.windowpath = [os.path.join(objdir, obj)
                           for obj in self.model['windows'].split()]
        self.viewdicts = {}
        views = [ent for ent in self.raysenders if ent.startswith('view')]
        if len(views) > 0:
            for view in views:
                if self.raysenders[view] is not None:
                    vdict = radutil.parse_vu(self.raysenders[view])
                    if 'vf' in vdict:
                        with open(vdict['vf']) as rdr:
                            vdict.update(radutil.parse_vu(rdr.read()))
                    self.viewdicts[view] = vdict
        if self.model['BSDF'] is not None:
            self.bsdfpath = [os.path.join(self.mtxdir, bsdf)
                             for bsdf in self.model['bsdf'].split()]
        if self.model['sunBSDF'] is not None:
            self.maccfspath = [os.path.join(objdir, obj)
                               for obj in self.model['sunBSDF'].split()]
        else:
            self.bsdfpath = None

    def assemble(self):
        with open(self.materialpath) as rdr:
            mat_prims = radutil.parse_primitive(rdr.readlines())
        black_mat = {'modifier':'void', 'type':'plastic',
                     'identifier':'black', 'str_args':'0',
                     'int_arg':'0', 'real_args':'5 0 0 0 0 0'}
        glow_mat = {'modifier':'void', 'type':'glow',
                    'identifier':'glowing', 'str_args':'0',
                    'int_arg':'0', 'real_args':'4 1 1 1 0'}
        with open(self.materialpath, 'a') as wtr:
            if black_mat not in mat_prims:
                wtr.write(radutil.put_primitive(black_mat))
            if glow_mat not in mat_prims:
                wtr.write(radutil.put_primitive(glow_mat))
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
            if self.bsdfpath is not None:
                self.bsdf[wname] = self.bsdfpath[idx]
        if self.raysenders['grid_surface'] is None:
            self.sensor_pts = None
        else:
            surface_path = os.path.join(
                self.filestrct['base'], self.filestrct['objects'],self.raysenders['grid_surface'])
            with open(surface_path) as rdr:
                prim = radutil.parse_primitive(rdr.readlines())
            self.sensor_pts = radutil.gen_grid(
                prim[0]['polygon'], float(self.raysenders['distance']),
                float(self.raysenders['spacing']), op=self.raysenders.getboolean('opposite'))
        try:
            with open(self.viewpath) as rdr:
                self.vu_dict = radutil.parse_vu(rdr.readlines()[0])
        except AttributeError:
            self.vu_dict = None


    def make_room(self):
        """Make a side-lit shoebox room."""
        theroom = room.Shoebox(float(self.dimensions['width']),
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
        sensor_grid = radutil.gen_grid(theroom.floor, self.raysenders['distance'],
                                       self.raysenders['spacing'],
                                       op=self.raysenders.getboolean('opposite'))
        nsensor = len(sensor_grid)
        return theroom, sensor_grid

    def get_wea(self):
        if (self.site['wea'] is not None) and (self.site['wea'] != ''):
            self.wea_path = os.path.join(
                self.filestrct['base'], self.filestrct['resources'], self.site['wea'])
            with open(self.wea_path) as rdr:
                raw = rdr.read()
            sec = raw.split('{os.linesep*2}')
            header = sec[0]
            lines = [l.split() for l in sec[1].splitines()]
            self.dts = [f"{int(l[0]):02d}{int(l[1]):02d}_{int(float(l[2])):02d}30" for l in lines]
        else:
            if self.site['zipcode'] is not None:
                epw = makesky.getEPW.from_zip(self.site['zipcode'])
                self.site['lat'], self.site['lon'] = str(epw.lat), str(epw.lon)
            elif None not in (self.site['latitude'], self.site['longitude']):
                epw = makesky.getEPW(self.site['latitude'], self.site['longitude'])
            else:
                raise NameError("Not site info defined")
            self.logger.info(f"Downloaded {epw.fname}")
            epw_path = os.path.join(
                self.filestrct['base'], self.filestrct['resources'], epw.fname)
            os.rename(epw.fname, epw_path)
            wea = makesky.epw2wea(
                epw=epw_path, dh=self.site.getboolean('daylight_hours_only'),
                sh=self.site['start_hour'], eh=self.site['end_hour'])
            self.wea_path = os.path.join(
                self.filestrct['base'], self.filestrct['resources'], radutil.basename(epw.fname) + '.wea')
            with open(self.wea_path, 'w') as wtr:
                wtr.write(wea.wea)
            self.dts = wea.dt_string
