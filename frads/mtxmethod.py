"""
Typical Radiance matrix-based simulation workflows
"""

import os
import copy
import shutil
import tempfile as tf
import logging
import subprocess as sp
from collections import namedtuple
from . import radutil, radmtx, makesky, mfacade

import pdb

logger = logging.getLogger('frads.mtxmethod')

cfg_template = {
    'SimulationControl':{
        'vmx_basis': 'kf', 'vmx_opt': '-ab 1 -ad 512',
        'fmx_basis': 'kf', 'fmx_opt': None,
        'smx_basis': 'r4', 'dmx_opt': '-ab 1 -ad 128 -c 2000',
        'dsmx_opt': '-ab 3 -ad 262144 -lw 1e-9', 'cdsmx_opt': '-ab 0', 'ray_count': 1,
        'pixel_jitter': .7, 'separate_direct': False, 'nprocess': 1, 'overwrite':True
    }, 'FileStructure':{
        'base': '', 'matrices': 'Matrices', 'results': 'Results',
        'objects': 'Objects', 'resources': 'Resources',
    }, 'Site':{
        'wea_path': None, 'latitude': None, 'longitude': None, 'zipcode': None,
        'daylight_hours_only': False, 'start_hour': None, 'end_hour': None,
        'orientation': 0,
    }, 'Model':{
        'material': None, 'windows': None, 'scene': None,
        'ncp_shade': None, 'BSDF': None, 'sunBSDF': None,
    }, 'Raysenders':{
        'view1': None, 'grid_surface': None, 'grid_height': None,
        'grid_spacing': None, 'grid_opposite': True,
    }}

pjoin = os.path.join


def mtxmult(*mtx):
    """Multiply matrices using rmtxop, convert to photopic, remove header."""
    cmd1 = ['dctimestep', '-od'] + list(mtx)
    cmd2 = ['rmtxop', '-fd', '-c', '47.4', '119.9', '11.6', '-', '-t']
    cmd3 = ['getinfo', '-']
    out1 = sp.Popen(cmd1, stdout=sp.PIPE)
    out2 = sp.Popen(cmd2, stdin=out1.stdout, stdout=sp.PIPE)
    out1.stdout.close()
    out3 = sp.Popen(cmd3, stdin=out2.stdout, stdout=sp.PIPE)
    out2.stdout.close()
    out = out3.communicate()[0]
    return out

def imgmult(*mtx, odir):
    """Image-based matrix multiplication using dctimestep."""
    cmd = ['dctimestep', '-o', pjoin(odir, '%04d.hdr')] + list(mtx)
    return cmd

#def get_avgskv(self):
    #radutil.sprun(f"gendaymtx -m {self.mf_sky} -A {self.wea} > {avgskyv}")




class MTXMethod:
    """Typical Radiance matrix-based simulation workflows
    Attributes:
        -processing for matrix-based simulation."""
    def __init__(self, config):
        self.logger = logging.getLogger('frads.mtxmethod.Prepare')
        self.config = namedtuple('config', config.keys())(**config)
        self.get_paths()
        self.assemble()
        self.get_wea()

    def get_paths(self, ):
        """Where are ?"""
        assert self.config.scene is not None, "No scene description"
        self.objdir = pjoin(self.config.base, self.config.objects)
        self.mtxdir = pjoin(self.config.base, self.config.matrices)
        self.resodir = pjoin(self.config.base, self.config.resources)
        self.resdir = pjoin(self.config.base, self.config.results)
        self.materialpath = pjoin(self.objdir, self.config.material)
        self.scenepath = [pjoin(self.objdir, obj) for obj in self.config.scene.split()]
        self.windowpath = [pjoin(self.objdir, obj) for obj in self.config.windows.split()]
        try:
            self.bsdfpath = [pjoin(self.resodir, bsdf) for bsdf in self.config.bsdf.split()]
            self.maccfspath = [pjoin(self.objdir, obj) for obj in self.config.sunBSDF.split()]
        except AttributeError:
            self.bsdfpath = None
            self.maccfspath = None
        self.envpath = [self.materialpath]
        self.envpath.extend(self.scenepath)

    def add_black_glow(self):
        """Add black and glow material to material file."""
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

    def assemble(self):
        """Assemble all the pieces together."""
        self.add_black_glow()
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
        if self.config.grid_surface is not None:
            surface_path = pjoin(self.objdir, self.config.grid_surface)
            with open(surface_path) as rdr:
                prim = radutil.parse_primitive(rdr.readlines())
            sensor_pts = radutil.gen_grid(
                prim[0]['polygon'], float(self.config.grid_height), float(self.config.grid_spacing))
            self.sndr_pts = radmtx.Sender.as_pts(pts_list=sensor_pts,
                                                 ray_cnt=int(self.config.ray_count))
        self.sndr_vus = {}
        views = [key for key, val in self.config._asdict().items()
                 if key.startswith('view') and val is not None]
        for view in views:
            vdict = radutil.parse_vu(self.config.view)
            if 'vf' in vdict:
                with open(vdict['vf']) as rdr:
                    vdict.update(radutil.parse_vu(rdr.read()))
            self.sndr_vus[view] = radmtx.Sender.as_view(
                vu_dict=vdict, ray_cnt=self.config.ray_count, xres=vdict['x'], yres=vdict['y'])
        self.rcvr_sky = radmtx.Receiver.as_sky(basis=self.config.smx_basis)

    def get_wea(self):
        """."""
        if self.config.wea_path is not None:
            self.wea_path = pjoin(self.resodir, self.config.wea_path)
            with open(self.wea_path) as rdr:
                raw = rdr.read()
            sec = raw.split('{os.linesep*2}')
            lines = [l.split() for l in sec[1].splitlines()]
            self.dts = [f"{int(l[0]):02d}{int(l[1]):02d}_{int(float(l[2])):02d}30" for l in lines]
        else:
            if self.config.zipcode is not None:
                epw = makesky.getEPW.from_zip(self.config.zipcode)
            elif None not in (self.config.latitude, self.config.longitude):
                epw = makesky.getEPW(self.config.latitude, -float(self.config.longitude))
            else:
                raise NameError("Not site info defined")
            self.logger.info("Downloaded : %s", epw.fname)
            epw_path = pjoin(self.resodir, epw.fname)
            try:
                os.rename(epw.fname, epw_path)
            except FileExistsError as fee:
                logger.info(fee)
            wea = makesky.epw2wea(
                epw=epw_path, dh=self.config.daylight_hours_only,
                sh=self.config.start_hour, eh=self.config.end_hour)
            self.wea_path = pjoin(self.resodir, radutil.basename(epw.fname) + '.wea')
            with open(self.wea_path, 'w') as wtr:
                wtr.write(wea.wea)
            self.dts = wea.dt_string

    def gen_smx(self, mfactor, onesun=False, direct=False):
        """Generate sky/sun matrix."""
        sun_only = ' -d' if direct else ''
        _five = ' -5 .533' if onesun else ''
        oname = radutil.basename(self.wea_path)
        cmd = f"gendaymtx -of -m {mfactor[-1]}{sun_only}{_five}".split()
        cmd.append(self.wea_path)
        res = radutil.spcheckout(cmd)
        if direct:
            if onesun:
                smxpath = pjoin(self.mtxdir, oname+'_d6.smx')
            else:
                smxpath = pjoin(self.mtxdir, oname+'_d.smx')
        else:
            smxpath = pjoin(self.mtxdir, oname+'.smx')
        with open(smxpath, 'wb') as wtr:
            wtr.write(res)
        return smxpath

    def prep_2phase_pt(self):
        """Prepare matrices two phase methods."""
        env = self.envpath + self.windowpath
        pdsmx = pjoin(self.mtxdir, 'pdsmx.mtx')
        if not os.path.isfile(pdsmx) or self.config.overwrite:
            res = radmtx.rfluxmtx(sender=self.sndr_pts, receiver=self.rcvr_sky,
                                  env=env, opt=self.config.dsmx_opt)
            with open(pdsmx, 'w') as wtr:
                wtr.write(res.decode())
        return pdsmx

    def prep_2phase_vu(self):
        """Generate image-based matrices if view defined."""
        env = self.envpath + self.windowpath
        vdsmx = {}
        for view in self.sndr_vus:
            vdsmx[view] = pjoin(self.mtxdir, f"vdsmx_{view}")
            radmtx.rfluxmtx(
                sender=self.sndr_vus[view], receiver=self.rcvr_sky,
                env=env, opt=self.config.dsmx_opt, out=vdsmx[view])
        return vdsmx

    def calc_2phase_pt(self, dsmx, smx):
        """."""
        res = mtxmult(dsmx, smx).splitlines()
        respath = pjoin(self.resdir, 'pdsmx.txt')
        with open(respath, 'w') as wtr:
            for idx, _ in enumerate(res):
                wtr.write(self.dts[idx] + '\t')
                wtr.write(res[idx].decode() + os.linesep)

    def calc_2phase_vu(self, dsmx, smx):
        """."""
        opath = pjoin(self.resdir, 'view2ph')
        for view in dsmx:
            radutil.sprun(imgmult(dsmx[view], smx, odir=opath))
            ofiles = [pjoin(opath, f) for f in sorted(os.listdir(opath))
                      if f.endswith('.hdr')]
            for idx, val in enumerate(ofiles):
                os.rename(val, pjoin(opath, self.dts[idx]+'.hdr'))

    def prep_3phase_dmx(self):
        """."""
        dmxs = {}
        for wname in self.window_prims:
            wndw_prim = self.window_prims[wname]
            self.logger.info("Generating daylight matrix for %s", wname)
            dmxs[wname] = pjoin(self.mtxdir, f'dmx_{wname}.mtx')
            sndr_wndw = radmtx.Sender.as_surface(
                prim_list=wndw_prim, basis=self.config.vmx_basis)
            dmx_res = radmtx.rfluxmtx(sender=sndr_wndw, receiver=self.rcvr_sky,
                       env=self.envpath, out=None, opt=self.config.dmx_opt)
            with open(dmxs[wname], 'wb') as wtr:
                wtr.write(dmx_res)
        return dmxs

    def prep_3phase_pt(self):
        """."""
        vmxs = {}
        rcvr_wndws = radmtx.Receiver(
            receiver='', basis=self.config.vmx_basis, modifier=None)
        for wname in self.window_prims:
            wndw_prim = self.window_prims[wname]
            vmxs[wname] = pjoin(self.mtxdir, f'pvmx_{wname}.mtx')
            rcvr_wndws += radmtx.Receiver.as_surface(
                prim_list=wndw_prim, basis=self.config.vmx_basis,
                offset=None, left=None, source='glow', out=vmxs[wname])
        self.logger.info("Generating view matrix for sensor point")
        radmtx.rfluxmtx(sender=self.sndr_pts, receiver=rcvr_wndws,
                        env=self.envpath, opt=self.config.vmx_opt, out=None)
        return vmxs

    def prep_3phase_vu(self):
        """."""
        vmxs = {}
        vrcvr_wndws = {}
        for view in self.sndr_vus:
            for wname in self.window_prims:
                vrcvr_wndws[view+wname] = radmtx.Receiver(
                    receiver='', basis=self.config.vmx_basis, modifier=None)
        for view in self.sndr_vus:
            for wname in self.window_prims:
                wndw_prim = self.window_prims[wname]
                vmxs[view+wname] = pjoin(
                    self.mtxdir, f'vvmx_{view}_{wname}', '%04d.hdr')
                radutil.mkdir_p(os.path.dirname(vmxs[view+wname]))
                vrcvr_wndws[view+wname] += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.config.vmx_basis, out=vmxs[view+wname])
        for view in self.sndr_vus:
            for wname in self.window_prims:
                self.logger.info("Generating view matrix for %s", view)
                radmtx.rfluxmtx(sender=self.sndr_vus[view], receiver=vrcvr_wndws[view+wname],
                                env=self.envpath, opt=self.config.vmx_opt, out=None)
        return vmxs

    def calc_3phase_pt(self, vmx, dmx, smx):
        """."""
        self.logger.info("Computing for point grid results")
        presl = []
        for wname in self.window_prims:
            _res = mtxmult(vmx[wname], self.bsdf[wname], dmx[wname], smx).splitlines()
            presl.append([map(float, l.strip().split('\t')) for l in _res.splitlines()])
        res = [[sum(tup) for tup in zip(*line)]for line in zip(*presl)]
        respath = pjoin(self.resdir, 'points3ph.txt')
        with open(respath, 'w') as wtr:
            for idx, val in enumerate(res):
                wtr.write(self.dts[idx] + ',')
                wtr.write(','.join(map(str, val)) + os.linesep)

    def calc_3phase_vu(self, vmx, dmx, smx):
        """."""
        for view in self.sndr_vus:
            self.logger.info("Computing for rendering results for %s", view)
            vresl = []
            for wname in self.window_prims:
                _vrespath = pjoin(self.resdir, f'{view}_{wname}')
                radutil.mkdir_p(_vrespath)
                cmd = imgmult(vmx[view+wname], self.bsdf[wname], dmx[wname], smx, odir=_vrespath)
                radutil.sprun(cmd)
                vresl.append(_vrespath)
            opath = pjoin(self.resdir, f'{view}_3ph')
            if len(vresl) > 1:
                [vresl.insert(i*2-1, '+') for i in range(1, len(vresl)+1)]
                radutil.pcombop(vresl, opath)
            else:
                if os.path.isdir(opath):
                    shutil.rmtree(opath)
                os.rename(vresl[0], opath)
            ofiles = [pjoin(opath, f) for f in sorted(os.listdir(opath)) if
                      f.endswith('.hdr')]
            [os.rename(ofiles[idx], pjoin(opath, self.dts[idx]+'.hdr'))
             for idx in range(len(ofiles))]

    def prep_4phase(self):
        """."""
        self.pvmxs = {}
        self.vvmxs = {}
        self.dmxs = {}
        self.fmxs = {}
        prcvr_wndws = radmtx.Receiver(receiver='', basis=self.vmx_basis, modifier=None)
        if len(self.sndr_views) > 0:
            vrcvr_wndws = {}
            for view in self.sndr_vus:
                vrcvr_wndws[view] = radmtx.Receiver(receiver='', basis=self.vmx_basis, modifier=None)
        port_prims = mfacade.genport(
            wpolys=wndw_prims, npolys=ncp_prims, depth=None, scale=None)
        mfacade.Genfmtx(win_polygons=wndw_polygon, port_prim=port_prims,
                        out=kwargs['o'], env=kwargs['env'], sbasis=kwargs['ss'],
                        rbasis=kwargs['rs'], opt=kwargs['opt'], refl=False,
                        forw=False, wrap=False)
        for wname in self.window_prims:
            wndw_prim = self.window_prims[wname]
            self.logger.info(f"Generating daylight matrix for {wname}")
            self.dmxs[wname] = pjoin(self.mtxdir, f'dmx_{wname}.mtx')
            sndr_wndw = radmtx.Sender.as_surface(
                prim_list=wndw_prim, basis=self.vmx_basis, offset=None)
            sndr_port = radmtx.Sender.as_surface(
                prim_list=port_prims, basis=self.fmx_basis, offset=None)
            radmtx.rfluxmtx(sender=sndr_port, receiver=self.rcvr_sky,
                       env=self.envpath, out=self.dmxs[wname], opt=self.dmx_opt)
            if self.sensor_pts is not None:
                self.pvmxs[wname] = pjoin(self.mtxdir, f'pvmx_{wname}.mtx')
                prcvr_wndws += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis,
                    offset=None, left=None, source='glow', out=self.pvmxs[wname])
            if len(self.sndr_vus) > 0:
                for view in self.sndr_vus:
                    self.vvmxs[view+wname] = pjoin(
                        self.mtxdir, f'vvmx_{view}_{wname}', '%04d.hdr')
                    radutil.mkdir_p(os.path.dirname(self.vvmxs[view+wname]))
                    vrcvr_wndws[view] += radmtx.Receiver.as_surface(
                        prim_list=wndw_prim, basis=self.vmx_basis,
                        offset=None, left=None, source='glow', out=self.vvmxs[view+wname])
        if self.sensor_pts is not None:
            self.logger.info("Generating view matrix for sensor point")
            radmtx.rfluxmtx(sender=self.sndr_pts, receiver=prcvr_wndws,
                            env=self.envpath, opt=self.vmx_opt, out=None)
        if len(self.sndr_vus) > 0:
            self.logger.info(f"Generating view matrix for {view}")
            for view in self.sndr_vus:
                radmtx.rfluxmtx(sender=self.sndr_vus[view], receiver=vrcvr_wndws[view],
                                env=self.envpath, opt=self.vmx_opt, out=None)

    def blacken_env(self):
        """."""
        blackened = copy.deepcopy(self.scene_prims)
        for prim in blackened:
            prim['modifier'] = 'black'
        self.blackenvpath = pjoin(self.objdir, 'blackened.rad')
        with open(self.blackenvpath, 'w') as wtr:
            for prim in blackened:
                wtr.write(radutil.put_primitive(prim))
        blackwindow = copy.deepcopy(self.window_prims)
        glowwindow = copy.deepcopy(self.window_prims)
        for wname in blackwindow:
            for prim in blackwindow[wname]:
                prim['modifier'] = 'black'
            for prim in glowwindow[wname]:
                prim['modifier'] = 'glowing'
        bwindow_path = pjoin(self.objdir, 'blackened_window.rad')
        gwindow_path = pjoin(self.objdir, 'glowing_window.rad')
        with open(bwindow_path, 'w') as wtr:
            for wndw in blackwindow:
                [wtr.write(radutil.put_primitive(prim)) for prim in blackwindow[wndw]]
        with open(gwindow_path, 'w') as wtr:
            for wndw in glowwindow:
                [wtr.write(radutil.put_primitive(prim)) for prim in glowwindow[wndw]]
        self.vmap_oct = pjoin(self.resodir, 'vmap.oct')
        self.cdmap_oct = pjoin(self.resodir, 'cdmap.oct')
        vmap = radutil.spcheckout(['oconv']+self.envpath+[bwindow_path])
        with open(self.vmap_oct, 'wb') as wtr: wtr.write(vmap)
        cdmap = radutil.spcheckout(['oconv']+self.envpath+[gwindow_path])
        with open(self.cdmap_oct, 'wb') as wtr: wtr.write(cdmap)


    def prep_5phase(self):
        """."""
        self.prep_3phase_dmx()
        smx_sun = gen_smx(self.config.cdsmx_basis, onesun=True, direct=True)
        self.blacken_env()
        rcvr_sun = radmtx.Receiver.as_sun(
            basis='r6', smx_path=self.smxd6path, window_paths=self.windowpath)
        sun_oct = pjoin(self.resodir, 'sun.oct')
        radmtx.rcvr_oct(rcvr_sun, [self.materialpath, self.blackenvpath], sun_oct)
        prcvr_wndws = radmtx.Receiver(receiver='', basis=self.vmx_basis, modifier=None)
        if len(self.sndr_vus) > 0:
            vdrcvr_wndws = {}
            for view in self.sndr_vus:
                vdrcvr_wndws[view] = radmtx.Receiver(receiver='', basis=self.vmx_basis, modifier=None)
        cdsenv = [self.materialpath, self.blackenvpath] + self.windowpath
        for wname in self.window_prims:
            wndw_prim = self.window_prims[wname]
            sndr_wndw = radmtx.Sender.as_surface(
                prim_list=wndw_prim, basis=self.vmx_basis, left=None, offset=None)
            self.dmxs[wname+'_d'] = pjoin(self.mtxdir, f'dmx_{wname}_d.mtx')
            self.logger.info(f"Generating direct daylight matrix for {wname}")
            dmx_res = radmtx.rfluxmtx(sender=sndr_wndw, receiver=self.rcvr_sky,
                            env=[self.materialpath, self.blackenvpath],
                            out=None, opt=self.dmx_opt+' -ab 0')
            with open(self.dmxs[wname+'_d'], 'wb') as wtr: wtr.write(dmx_res)
            if self.sensor_pts is not None:
                self.pvmxs[wname+'_d'] = pjoin(self.mtxdir, f'pvmx_{wname}_d.mtx')
                prcvr_wndws += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis,
                    offset=None, left=None, source='glow', out=self.pvmxs[wname+'_d'])
            for vu in self.sndr_vus:
                self.vvmxs[view+wname+'_d'] = pjoin(
                    self.mtxdir, f'vvmx_{view}_{wname}_d', '%04d.hdr')
                radutil.mkdir_p(os.path.dirname(self.vvmxs[view+wname+'_d']))
                vdrcvr_wndws[view] += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis,
                    offset=None, left=None, source='glow',
                    out=self.vvmxs[view+wname+'_d'])
        if self.sensor_pts is not None:
            self.logger.info(f"Generating direct view matrix for sensor grid")
            radmtx.rfluxmtx(sender=self.sndr_pts, receiver=prcvr_wndws,
                            env=[self.materialpath, self.blackenvpath],
                            out=None, opt=self.vmx_opt+' -ab 1')
            self.logger.info(f"Generating direct sun matrix for sensor_grid")
            self.pcdsmx = pjoin(self.mtxdir, 'pcdsmx.mtx')
            radmtx.rcontrib(sender=self.sndr_pts, modifier=rcvr_sun.modifier,
                            octree=sun_oct, out=self.pcdsmx, opt=self.cdsmx_opt)
        self.vcdfmx = {}
        self.vcdrmx = {}
        self.vmap_paths = {}
        self.cdmap_paths = {}
        for view in self.sndr_vus:
            self.vmap_paths[view] = pjoin(self.mtxdir, f'vmap_{view}.hdr')
            self.cdmap_paths[view] = pjoin(self.mtxdir, f'cdmap_{view}.hdr')
            vdict = self.viewdicts[view]
            vdict.pop('c', None)
            vdict.pop('pj', None)
            view_str = radutil.opt2str(vdict)
            cmd = ['rpict'] + view_str.split() + ['-ps', '1', '-ab', '0', '-av']
            cmd.extend(['.31831', '.31831', '.31831', self.vmap_oct])
            vmap = radutil.spcheckout(cmd)
            with open(self.vmap_paths[view], 'wb') as wtr: wtr.write(vmap)
            cmd[-1] = self.cdmap_oct
            cdmap = radutil.spcheckout(cmd)
            with open(self.cdmap_paths[view], 'wb') as wtr: wtr.write(cdmap)
            self.logger.info(f"Generating direct view matrix for {view}")
            radmtx.rfluxmtx(sender=self.sndr_vus[view], receiver=vdrcvr_wndws[view],
                            env=[self.materialpath, self.blackenvpath],
                            opt=self.config.vmx_opt+' -ab 0 -i', out=None)
            self.vcdfmx[view] = pjoin(self.mtxdir, f'vcdfmx_{view}')
            self.vcdrmx[view] = pjoin(self.mtxdir, f'vcdrmx_{view}')
            self.logger.info(f"Generating direct sun r matrix for {view}")
            radmtx.rcontrib(sender=self.sndr_vus[view], modifier=rcvr_sun.modifier,
                            octree=sun_oct, out=self.vcdrmx[view], opt=self.cdsmx_opt+' -i')
            self.logger.info(f"Generating direct sun f matrix for {view}")
            radmtx.rcontrib(sender=self.sndr_vus[view], modifier=rcvr_sun.modifier,
                            octree=sun_oct, out=self.vcdfmx[view], opt=self.cdsmx_opt)

    def calc_5phase_pt(self):
        """."""
        self.gen_smx(self.dmx_basis)
        if self.sensor_pts is not None:
            presl = []
            pdresl = []
            mult_cds = radutil.spcheckout(self.mtxmult(self.pcdsmx, self.smxd6path))
            prescd = [list(map(float, l.strip().split('\t')))
                     for l in mult_cds.splitlines()]
            for wname in self.window_prims:
                _res = mtxmult(self.pvmxs[wname], self.bsdf[wname], self.dmxs[wname], self.smxpath)
                _resd = mtxmult(self.pvmxs[wname+'_d'], self.bsdf[wname], self.dmxs[wname+'_d'], self.smxdpath)
                presl.append([map(float, l.strip().split('\t')) for l in _res.splitlines()])
                pdresl.append([map(float, l.strip().split('\t')) for l in _resd.splitlines()])
            pres3 = [[sum(tup) for tup in zip(*line)]for line in zip(*presl)]
            pres3d = [[sum(tup) for tup in zip(*line)]for line in zip(*pdresl)]
            res = [[x-y+z for x,y,z in zip(a,b,c)] for a,b,c in zip(pres3, pres3d, prescd)]
            respath = pjoin(self.resdir, 'points5ph.txt')
            with open(respath, 'w') as wtr:
                for idx in range(len(res)):
                    wtr.write(self.dts[idx] + ',')
                    wtr.write(','.join(map(str, res[idx])) + os.linesep)

    def calc_5phase_vu(self):
        """."""
        for view in self.sndr_vus:
            self.logger.info(f"Computing for rendering results for {view}")
            vresl = []
            vdresl = []
            vrescdr = tf.mkdtemp(dir=self.td)
            vrescdf = tf.mkdtemp(dir=self.td)
            vrescd = tf.mkdtemp(dir=self.td)
            cmds = []
            cmds.append(self.imgmult(pjoin(self.vcdrmx[view], '%04d.hdr'),
                                     self.smxd6path, odir=vrescdr))
            cmds.append(self.imgmult(pjoin(self.vcdfmx[view], '%04d.hdr'),
                                     self.smxd6path, odir=vrescdf))
            for wname in self.window_prims:
                _vrespath = tf.mkdtemp(dir=self.td)
                _vdrespath = tf.mkdtemp(dir=self.td)
                cmds.append(self.imgmult(self.vvmxs[view+wname], self.bsdf[wname],
                             self.dmxs[wname], self.smxpath, odir=_vrespath))
                cmds.append(self.imgmult(self.vvmxs[view+wname+'_d'], self.bsdf[wname],
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
            radutil.pcombop([res3di, '*', self.vmap_paths[view]], res3d)
            radutil.pcombop([vrescdr, '*', self.cdmap_paths[view], '+', vrescdf], vrescd)
            opath = pjoin(self.resdir, f"{view}_5ph")
            if os.path.isdir(opath): shutil.rmtree(opath)
            radutil.pcombop([res3, '-', res3d, '+', vrescd], opath)
            ofiles = [pjoin(opath, f) for f in sorted(os.listdir(opath)) if
                      f.endswith('.hdr')]
            [os.rename(ofiles[idx], pjoin(opath, self.dts[idx]+'.hdr'))
             for idx in range(len(ofiles))]
            self.logger.info(f"Done computing for {view}")
        shutil.rmtree(self.td)
