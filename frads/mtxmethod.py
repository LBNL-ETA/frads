"""
Typical Radiance matrix-based simulation workflows
"""

from collections import namedtuple
import copy
import logging
import multiprocessing as mp
import os
import shutil
import subprocess as sp
import tempfile as tf
from frads import radutil, radmtx, makesky, mfacade

logger = logging.getLogger('frads.mtxmethod')

cfg_template = {
    'vmx_basis': 'kf', 'vmx_opt': '-ab 5 -ad 65536 -lw 1e-8',
    'fmx_basis': 'kf', 'fmx_opt': None,
    'smx_basis': 'r4', 'dmx_opt': '-ab 2 -ad 128 -c 2000',
    'dsmx_opt': '-ab 2 -ad 64 -lw 1e-4',
    'cdsmx_opt': '-ab 0', 'ray_count': 1,
    'pixel_jitter': .7, 'separate_direct': False,
    'nprocess': 1, 'overwrite': False, 'method':None,
    'base': '', 'matrices': 'Matrices', 'results': 'Results',
    'objects': 'Objects', 'resources': 'Resources',
    'wea_path': None, 'latitude': None, 'longitude': None, 'zipcode': None,
    'daylight_hours_only': False, 'start_hour': None, 'end_hour': None,
    'orientation': 0,
    'material': None, 'windows': None, 'scene': None,
    'ncp_shade': None, 'bsdf': None, 'dbsdf': None,
    'view1': None, 'grid_surface': None, 'grid_height': .76,
    'grid_spacing': .6, 'grid_opposite': True,
}

pjoin = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile


def mtxmult(*mtx):
    """Multiply matrices using rmtxop, convert to photopic, remove header."""
    cmd1 = ['dctimestep', '-od'] + list(mtx)
    cmd2 = ['rmtxop', '-fa', '-c', '47.4', '119.9', '11.6', '-', '-t']
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
    radutil.mkdir_p(odir)
    cmd = ['dctimestep', '-oc', '-o', pjoin(odir, '%04d.hdr')] + list(mtx)
    return cmd


def two_phase(_setup, smx):
    """Two-phase simulation workflow."""
    pdsmx = _setup.prep_2phase_pt()
    vdsmx = _setup.prep_2phase_vu()
    _setup.calc_2phase_pt(pdsmx, smx)
    _setup.calc_2phase_vu(vdsmx, smx)


def three_phase(_setup, smx, direct=False):
    """3/5-phase simulation workflow."""
    pvmxs = _setup.view_matrix_pt()
    vvmxs = _setup.view_matrix_vu()
    dmxs = _setup.daylight_matrix(_setup.window_prims)
    if direct:
        smxd = _setup.gen_smx(_setup.wea_path, _setup.config.smx_basis, _setup.mtxdir, direct=True)
        smx_sun = _setup.gen_smx(_setup.wea_path, _setup.config.cdsmx_basis, _setup.mtxdir, onesun=True, direct=True)
        _setup.blacken_env()
        pcdsmx = _setup.direct_sun_matrix_pt(smx_sun)
        vcdfmx, vcdrmx, vmap_paths, cdmap_paths = _setup.direct_sun_matrix_vu(smx_sun)
        dmxsd = _setup.daylight_matrix(_setup.window_prims, direct=True)
        pvmxsd = _setup.view_matrix_pt(direct=True)
        vvmxsd = _setup.view_matrix_vu(direct=True)
        _setup.calc_5phase_pt(pvmxs, pvmxsd, dmxs, dmxsd, pcdsmx, smx, smxd, smx_sun)
        _setup.calc_5phase_vu(vvmxs, vvmxsd, dmxs, dmxsd, vcdfmx,
                              vcdrmx, vmap_paths, cdmap_paths, smx, smxd, smx_sun)
    else:
        _setup.calc_3phase_pt(pvmxs, dmxs, smx)
        _setup.calc_3phase_vu(vvmxs, dmxs, smx)

def four_phase(_setup, smx, direct=False):
    """Four-phase simulation workflow."""
    pvmxs = _setup.view_matrix_pt()
    vvmxs = _setup.view_matrix_vu()
    fmxs = _setup.facade_matrix()
    dmxs = _setup.daylight_matrix(_setup.port_prims)
    if direct:
        smxd = _setup.gen_smx(_setup.config.smx_basis, direct=True)
        smx_sun = _setup.gen_smx(_setup.config.cdsmx_basis, onesun=True, direct=True)
        _setup.blacken_env()
        pcdsmx = _setup.direct_sun_matrix_pt(smx_sun)
        vcdfmx, vcdrmx, vmap_paths, cdmap_paths = _setup.direct_sun_matrix_vu(smx_sun)
        dmxsd = _setup.daylight_matrix(_setup.window_prims, direct=True)
        fmxsd = _setup.facade_matrix(direct=True)
        pvmxsd = _setup.view_matrix_pt(direct=True)
        vvmxsd = _setup.view_matrix_vu(direct=True)
        _setup.calc_6phase_pt(pvmxs, pvmxsd, fmxs, fmxsd, dmxs, dmxsd, pcdsmx, smx, smxd, smx_sun)
        _setup.calc_6phase_vu(vvmxs, vvmxsd, fmxs, fmxsd, dmxs, dmxsd, vcdfmx,
                              vcdrmx, vmap_paths, cdmap_paths, smx, smxd, smx_sun)
    else:
        _setup.calc_4phase_pt(pvmxs, fmxs, dmxs, smx)
        _setup.calc_4phase_vu(vvmxs, fmxs, dmxs, smx)


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
        if self.config.scene is None:
            raise Exception("Scene files not provided")
        self.scenepath = [pjoin(self.objdir, obj)
                          for obj in self.config.scene.split()]
        if self.config.windows is None:
            self.windowpath = []
        else:
            self.windowpath = [pjoin(self.objdir, obj)
                               for obj in self.config.windows.split()]
        self.maccfspath = None if self.config.dbsdf is None else \
            [pjoin(self.objdir, obj) for obj in self.config.dbsdf.split()]
        self.bsdfpath = None if self.config.bsdf is None else \
            [pjoin(self.resodir, bsdf) for bsdf in self.config.bsdf.split()]
        self.envpath = [self.materialpath]
        self.envpath.extend(self.scenepath)
        if self.config.ncp_shade is not None:
            ncppaths = self.config.ncp_shade.split()
            if len(ncppaths) > 1:
                self.ncppath = [pjoin(self.objdir, obj) for obj in ncppaths]
            else:
                self.envpath.extend(ncppaths)

    def _add_black_glow(self):
        """Add black and glow material to material file."""
        with open(self.materialpath) as rdr:
            mat_prims = radutil.parse_primitive(rdr.readlines())
        black_mat = {'modifier': 'void', 'type': 'plastic',
                     'identifier': 'black', 'str_args': '0',
                     'int_arg': '0', 'real_args': '5 0 0 0 0 0'}
        glow_mat = {'modifier': 'void', 'type': 'glow',
                    'identifier': 'glowing', 'str_args': '0',
                    'int_arg': '0', 'real_args': '4 1 1 1 0'}
        with open(self.materialpath, 'a') as wtr:
            if black_mat not in mat_prims:
                wtr.write(radutil.put_primitive(black_mat))
            if glow_mat not in mat_prims:
                wtr.write(radutil.put_primitive(glow_mat))

    def assemble(self):
        """Assemble all the pieces together."""
        self._add_black_glow()
        self.scene_prims = []
        for spath in self.scenepath:
            with open(spath) as rdr:
                self.scene_prims.extend(
                    radutil.parse_primitive(rdr.readlines()))
        self.window_prims = {}
        self.bsdf = {}
        for idx, windowpath in enumerate(self.windowpath):
            wname = radutil.basename(windowpath)
            with open(windowpath) as rdr:
                self.window_prims[wname] = radutil.parse_primitive(
                    rdr.readlines())
            if self.bsdfpath is not None:
                self.bsdf[wname] = self.bsdfpath[idx]
        self.sndr_pts = None
        if self.config.grid_surface is not None:
            surface_path = pjoin(self.objdir, self.config.grid_surface)
            with open(surface_path) as rdr:
                prim = radutil.parse_primitive(rdr.readlines())
            sensor_pts = radutil.gen_grid(
                prim[0]['polygon'], float(self.config.grid_height), float(self.config.grid_spacing))
            self.sndr_pts = radmtx.Sender.as_pts(
                pts_list=sensor_pts, ray_cnt=int(self.config.ray_count))
        self.sndr_vus = {}
        views = [key for key, val in self.config._asdict().items()
                 if key.startswith('view') and val is not None]
        self.viewdicts = {}
        for view in views:
            vdict = radutil.parse_vu(self.config.__getattribute__(view))
            if 'vf' in vdict:
                with open(vdict['vf']) as rdr:
                    vdict.update(radutil.parse_vu(rdr.read()))
            self.viewdicts[view] = vdict
            self.sndr_vus[view] = radmtx.Sender.as_view(
                vu_dict=vdict, ray_cnt=int(self.config.ray_count), xres=vdict['x'], yres=vdict['y'])
        self.rcvr_sky = radmtx.Receiver.as_sky(basis=self.config.smx_basis)

    def get_wea(self):
        """Obtain and prepare weather file data."""
        if self.config.wea_path is not None:
            self.logger.info('Using user specified .wea file.')
            self.wea_path = pjoin(self.resodir, self.config.wea_path)
            with open(self.wea_path) as rdr:
                raw = rdr.read()
            sec = raw.split('\n\n')
            lines = [l.split() for l in sec[1].splitlines()]
            self.datetime_stamps = []
            for line in lines:
                month = int(line[0])
                day = int(line[1])
                hours = float(line[2])
                hour = int(hours)
                minutes = 60 * (hours - hour)
                self.datetime_stamps.append('%02d%02d_%02d%02d'%(month, day, hour, minutes))
        else:
            if self.config.zipcode is not None:
                self.logger.info('Downloading EPW file using zipcode.')
                epw = makesky.getEPW.from_zip(self.config.zipcode)
            elif None not in (self.config.latitude, self.config.longitude):
                self.logger.info('Downloading EPW file using lat&lon.')
                epw = makesky.getEPW(self.config.latitude,
                                     float(self.config.longitude))
            else:
                raise NameError("Not site info defined")
            self.logger.info("Downloaded : %s", epw.fname)
            epw_path = pjoin(self.resodir, epw.fname)
            try:
                os.rename(epw.fname, epw_path)
            except FileExistsError as fee:
                self.logger.info(fee)
            self.logger.info('Converting EPW to a .wea file')
            wea = makesky.epw2wea(
                epw=epw_path, dh=self.config.daylight_hours_only,
                sh=self.config.start_hour, eh=self.config.end_hour)
            self.wea_path = pjoin(
                self.resodir, radutil.basename(epw.fname) + '.wea')
            with open(self.wea_path, 'w') as wtr:
                wtr.write(wea.wea)
            self.datetime_stamps = wea.dt_string

    @staticmethod
    def gen_smx(wea_path, mfactor, outdir, onesun=False, direct=False):
        """Generate sky/sun matrix."""
        sun_only = ' -d' if direct else ''
        _five = ' -5 .533' if onesun else ''
        oname = radutil.basename(wea_path)
        cmd = f"gendaymtx -of -m {mfactor[-1]}{sun_only}{_five}".split()
        cmd.append(wea_path)
        logger.info('Generating sku/sun matrix using command')
        logger.info(' '.join(cmd))
        res = radutil.spcheckout(cmd)
        if direct:
            if onesun:
                smxpath = pjoin(outdir, oname+'_d6.smx')
            else:
                smxpath = pjoin(outdir, oname+'_d.smx')
        else:
            smxpath = pjoin(outdir, oname+'.smx')
        with open(smxpath, 'wb') as wtr:
            wtr.write(res)
        return smxpath

    def prep_2phase_pt(self):
        """Prepare matrices two phase methods."""
        self.logger.info('Computing for 2-phase sensor point matrices...')
        env = self.envpath + self.windowpath
        pdsmx = pjoin(self.mtxdir, 'pdsmx.mtx')
        if not isfile(pdsmx) or self.config.overwrite:
            res = radmtx.rfluxmtx(sender=self.sndr_pts, receiver=self.rcvr_sky,
                                  env=env, opt=self.config.dsmx_opt)
            with open(pdsmx, 'wb') as wtr:
                wtr.write(res)
        return pdsmx

    def prep_2phase_vu(self):
        """Generate image-based matrices if view defined."""
        self.logger.info("Computing for image-based 2-phase matrices...")
        env = self.envpath + self.windowpath
        vdsmx = {}
        for view in self.sndr_vus:
            vdsmx[view] = pjoin(self.mtxdir, f"vdsmx_{view}")
            if not isdir(vdsmx[view]) or self.config.overwrite:
                self.logger.info("Generating for "+ view)
                radmtx.rfluxmtx(
                    sender=self.sndr_vus[view], receiver=self.rcvr_sky,
                    env=env, opt=self.config.dsmx_opt, out=vdsmx[view])
        return vdsmx

    def calc_2phase_pt(self, dsmx, smx):
        """."""
        self.logger.info("Computing for 2-phase sensor grid results.")
        res = mtxmult(dsmx, smx).splitlines()
        respath = pjoin(self.resdir, 'pdsmx.txt')
        with open(respath, 'w') as wtr:
            for idx, _ in enumerate(res):
                wtr.write(self.datetime_stamps[idx] + '\t')
                wtr.write(res[idx].decode() + '\n')

    def calc_2phase_vu(self, dsmx, smx):
        """."""
        self.logger.info("Computing for 2-phase image-based results")
        opath = pjoin(self.resdir, 'view2ph')
        if os.path.isdir(opath):
            shutil.rmtree(opath)
        for view in dsmx:
            radutil.sprun(
                imgmult(pjoin(dsmx[view], '%04d.hdr'), smx, odir=opath))
            ofiles = [pjoin(opath, f) for f in sorted(os.listdir(opath))
                      if f.endswith('.hdr')]
            for idx, val in enumerate(ofiles):
                os.rename(val, pjoin(opath, self.datetime_stamps[idx]+'.hdr'))

    def daylight_matrix(self, sender_prims: dict, direct=False):
        """."""
        self.logger.info("Computing daylight matrix...")
        dmxs = {}
        _opt = self.config.dmx_opt
        _env = self.envpath
        if direct:
            _opt += ' -ab 0'
            _env = [self.materialpath, self.blackenvpath]
        for sname in sender_prims:
            _name = sname
            if direct:
                _name += '_d'
            dmxs[sname] = pjoin(self.mtxdir, f'dmx_{_name}.mtx')
            window_prim = sender_prims[sname]
            sndr_window = radmtx.Sender.as_surface(
                prim_list=window_prim, basis=self.config.vmx_basis)
            if not isfile(dmxs[sname]) or self.config.overwrite:
                self.logger.info("Generating daylight matrix for %s", _name)
                dmx_res = radmtx.rfluxmtx(sender=sndr_window, receiver=self.rcvr_sky,
                                          env=_env, out=None, opt=_opt)
                with open(dmxs[sname], 'wb') as wtr:
                    wtr.write(dmx_res)
        return dmxs

    def facade_matrix(self, direct=False):
        self.logger.info("Computing facade matrix...")
        fmxs = {}
        _opt = self.config.dmx_opt
        _env = self.envpath
        if direct:
            _opt += ' -ab 0'
            _env = [self.materialpath, self.blackenvpath]
        ncp_prims = {}
        for ncppath in self.ncppath:
            name = radutil.basename(ncppath)
            with open(ncppath) as rdr:
                ncp_prims[name] = radutil.parse_primitive(rdr.readlines())
        all_ncp_prims = [prim for key, prim in ncp_prims.items()]
        all_window_prims = [prim for key, prim in self.window_prims.items()]
        port_prims = mfacade.genport(wpolys=all_window_prims, npolys=all_ncp_prims,
                                     depth=None, scale=None)
        port_rcvr = radmtx.Receiver.as_surface(
            prim_list=port_prims, basis=self.config.fmx_basis, out=None)
        for wname in self.window_prims:
            _name = wname + '_d' if direct else wname
            fmxs[wname] = pjoin(self.mtxdir, f'fmx_{_name}.mtx')
            window_prim = self.window_prims[wname]
            sndr_window = radmtx.Sender.as_surface(
                prim_list=window_prim, basis=self.config.fmx_basis)
            if not isfile(fmxs[wname]) or self.config.overwrite:
                self.logger.info("Generating facade matrix for %s", _name)
                fmx_res = radmtx.rfluxmtx(sender=sndr_window, receiver=port_rcvr,
                                          env=_env, out=None, opt=_opt)
                with open(fmxs[wname], 'wb') as wtr:
                    wtr.write(fmx_res)
        return fmxs

    def view_matrix_pt(self, direct=False):
        """."""
        _opt = self.config.vmx_opt
        _env = self.envpath
        if direct:
            self.logger.info("Computing direct view matrix for sensor grid:")
            _opt += ' -ab 1'
            _env = [self.materialpath, self.blackenvpath]
        else:
            self.logger.info("Computing view matrix for sensor grid:")
        vmxs = {}
        rcvr_windows = radmtx.Receiver(
            receiver='', basis=self.config.vmx_basis, modifier=None)
        for wname in self.window_prims:
            window_prim = self.window_prims[wname]
            _name = wname
            if direct:
                _name += '_d'
            vmxs[wname] = pjoin(self.mtxdir, f'pvmx_{_name}.mtx')
            rcvr_windows += radmtx.Receiver.as_surface(
                prim_list=window_prim, basis=self.config.vmx_basis,
                offset=None, left=None, source='glow', out=vmxs[wname])
        if not all([isfile(f) for f in vmxs.values()]) or self.config.overwrite:
            self.logger.info(f"Generating using rfluxmtx")
            radmtx.rfluxmtx(sender=self.sndr_pts, receiver=rcvr_windows,
                            env=_env, opt=_opt, out=None)
        return vmxs

    def view_matrix_vu(self, direct=False):
        """Prepare matrices using three-phase methods."""
        self.logger.info("Computing image-based view matrix:")
        _opt = self.config.vmx_opt
        _env = self.envpath
        if direct:
            _opt += ' -i -ab 1'
            _env = [self.materialpath, self.blackenvpath]
        vmxs = {}
        vrcvr_windows = {}
        for view in self.sndr_vus:
            for wname in self.window_prims:
                vrcvr_windows[view+wname] = radmtx.Receiver(
                    receiver='', basis=self.config.vmx_basis, modifier=None)
        for view in self.sndr_vus:
            for wname, window_prim in self.window_prims.items():
                _name = view + wname
                if direct:
                    _name += '_d'
                vmxs[view+wname] = pjoin(
                    self.mtxdir, f'vvmx_{_name}', '%04d.hdr')
                vrcvr_windows[view+wname] += radmtx.Receiver.as_surface(
                    prim_list=window_prim, basis=self.config.vmx_basis, out=vmxs[view+wname])
        for view in self.sndr_vus:
            for wname in self.window_prims:
                if not isdir(vmxs[view+wname][:-8]) or self.config.overwrite:
                    self.logger.info("Generating for %s", view)
                    radutil.mkdir_p(os.path.dirname(vmxs[view+wname]))
                    radmtx.rfluxmtx(sender=self.sndr_vus[view], receiver=vrcvr_windows[view+wname],
                                    env=_env, opt=_opt, out=None)
        return vmxs

    def calc_3phase_pt(self, vmx, dmx, smx):
        """."""
        self.logger.info("Computing for 3-phase sensor grid results")
        presl = []
        for wname in self.window_prims:
            _res = mtxmult(vmx[wname], self.bsdf[wname],
                           dmx[wname], smx).splitlines()
            presl.append([map(float, l.decode().strip().split('\t'))
                          for l in _res])
        res = [[sum(tup) for tup in zip(*line)]for line in zip(*presl)]
        respath = pjoin(self.resdir, 'points3ph.txt')
        with open(respath, 'w') as wtr:
            for idx, val in enumerate(res):
                wtr.write(self.datetime_stamps[idx] + ',')
                wtr.write(','.join(map(str, val)) + '\n')

    def calc_3phase_vu(self, vmx, dmx, smx):
        """."""
        self.logger.info("Computing for 3-phase image-based results:")
        for view in self.sndr_vus:
            opath = pjoin(self.resdir, f'{view}_3phm')
            if not isdir(opath) or self.config.overwrite:
                self.logger.info("for %s", view)
                vresl = []
                for wname in self.window_prims:
                    _vrespath = pjoin(self.resdir, f'{view}_{wname}')
                    radutil.mkdir_p(_vrespath)
                    cmd = imgmult(vmx[view+wname], self.bsdf[wname],
                                  dmx[wname], smx, odir=_vrespath)
                    radutil.sprun(cmd)
                    vresl.append(_vrespath)
                if len(vresl) > 1:
                    for i in range(1, len(vresl)):
                        vresl.insert(i*2-1, '+')
                    radutil.pcombop(vresl, opath)
                else:
                    if os.path.isdir(opath):
                        shutil.rmtree(opath)
                    os.rename(vresl[0], opath)
                ofiles = [pjoin(opath, f) for f in sorted(os.listdir(opath)) if
                          f.endswith('.hdr')]
                for idx, ofile in enumerate(ofiles):
                    os.rename(ofile, pjoin(opath, self.datetime_stamps[idx]+'.hdr'))

    def prep_4phase_pt(self, direct=False):
        """Prepare matrices using four-phase methods for point-based calculation."""
        dmxs = {}
        fmxs = {}
        _opt = self.config.fmx_opt
        _env = self.envpath
        if direct:
            # need to add ncp path
            _env = [self.materialpath, self.blacken_env]
            _opt += ' -ab 0'
        ncp_prims = None
        for wname in self.window_prims:
            window_prim = self.window_prims[wname]
            if direct:
                _name = wname + '_d'
            fmxs[wname] = pjoin(self.mtxdir, 'fmx_{_name}.mtx')
            port_prims = mfacade.genport(
                wpolys=window_prim, npolys=ncp_prims, depth=None, scale=None)
            mfacade.Genfmtx(win_polygons=window_prim['polygon'], port_prim=port_prims,
                            out=fmxs[wname], env=_env, sbasis=self.config.vmx_basis,
                            rbasis=self.config.fmx_basis, opt=_opt, refl=False,
                            forw=False, wrap=False)
            self.logger.info(f"Generating daylight matrix for {wname}")
            dmxs[wname] = pjoin(self.mtxdir, f'dmx_{wname}.mtx')
            sndr_port = radmtx.Sender.as_surface(
                prim_list=port_prims, basis=self.config.fmx_basis, offset=None)
            radmtx.rfluxmtx(sender=sndr_port, receiver=self.rcvr_sky,
                            env=_env, out=dmxs[wname], opt=self.config.dmx_opt)
        return fmxs, dmxs

    def prep_4phase_vu(self):
        """."""
        vvmxs = {}
        dmxs = {}
        fmxs = {}
        prcvr_windows = radmtx.Receiver(
            receiver='', basis=self.vmx_basis, modifier=None)
        if len(self.sndr_views) > 0:
            vrcvr_windows = {}
            for view in self.sndr_vus:
                vrcvr_windows[view] = radmtx.Receiver(
                    receiver='', basis=self.vmx_basis, modifier=None)
        port_prims = mfacade.genport(
            wpolys=window_prims, npolys=ncp_prims, depth=None, scale=None)
        mfacade.Genfmtx(win_polygons=window_polygon, port_prim=port_prims,
                        out=kwargs['o'], env=kwargs['env'], sbasis=kwargs['ss'],
                        rbasis=kwargs['rs'], opt=kwargs['opt'], refl=False,
                        forw=False, wrap=False)
        for wname in self.window_prims:
            window_prim = self.window_prims[wname]
            self.logger.info(f"Generating daylight matrix for {wname}")
            dmxs[wname] = pjoin(self.mtxdir, f'dmx_{wname}.mtx')
            sndr_window = radmtx.Sender.as_surface(
                prim_list=window_prim, basis=self.vmx_basis, offset=None)
            sndr_port = radmtx.Sender.as_surface(
                prim_list=port_prims, basis=self.fmx_basis, offset=None)
            radmtx.rfluxmtx(sender=sndr_port, receiver=self.rcvr_sky,
                            env=self.envpath, out=dmxs[wname], opt=self.dmx_opt)
            for view in self.sndr_vus:
                vvmxs[view+wname] = pjoin(
                    self.mtxdir, f'vvmx_{view}_{wname}', '%04d.hdr')
                radutil.mkdir_p(os.path.dirname(vvmxs[view+wname]))
                vrcvr_windows[view] += radmtx.Receiver.as_surface(
                    prim_list=window_prim, basis=self.vmx_basis,
                    offset=None, left=None, source='glow', out=vvmxs[view+wname])
        self.logger.info(f"Generating view matrix for {view}")
        for view in self.sndr_vus:
            radmtx.rfluxmtx(sender=self.sndr_vus[view], receiver=vrcvr_windows[view],
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
            for window in blackwindow:
                [wtr.write(radutil.put_primitive(prim))
                 for prim in blackwindow[window]]
        with open(gwindow_path, 'w') as wtr:
            for window in glowwindow:
                [wtr.write(radutil.put_primitive(prim))
                 for prim in glowwindow[window]]
        self.vmap_oct = pjoin(self.resodir, 'vmap.oct')
        self.cdmap_oct = pjoin(self.resodir, 'cdmap.oct')
        vmap = radutil.spcheckout(['oconv']+self.envpath+[bwindow_path])
        with open(self.vmap_oct, 'wb') as wtr:
            wtr.write(vmap)
        cdmap = radutil.spcheckout(['oconv']+self.envpath+[gwindow_path])
        with open(self.cdmap_oct, 'wb') as wtr:
            wtr.write(cdmap)

    def direct_sun_matrix_pt(self, smx_path):
        """."""
        self.logger.info(f"Direct sun matrix for sensor grid")
        rcvr_sun = radmtx.Receiver.as_sun(
            basis='r6', smx_path=smx_path, window_paths=self.windowpath)
        sun_oct = pjoin(self.resodir, 'sun.oct')
        cdsenv = [self.materialpath, self.blackenvpath] + self.windowpath
        radmtx.rcvr_oct(rcvr_sun, cdsenv, sun_oct)
        pcdsmx = pjoin(self.mtxdir, 'pcdsmx.mtx')
        if not isfile(pcdsmx) or self.config.overwrite:
            self.logger.info(f"Generating using rcontrib...")
            radmtx.rcontrib(sender=self.sndr_pts, modifier=rcvr_sun.modifier,
                            octree=sun_oct, out=pcdsmx, opt=self.config.cdsmx_opt)
        return pcdsmx

    def direct_sun_matrix_vu(self, smx_path):
        """."""
        self.logger.info(f"Generating image-based direct sun matrix")
        rcvr_sun = radmtx.Receiver.as_sun(
            basis='r6', smx_path=smx_path, window_paths=self.windowpath)
        mod_names = [str(int(l[3:])-1) for l in rcvr_sun.modifier.splitlines()]
        sun_oct = pjoin(self.resodir, 'sun.oct')
        cdsenv = [self.materialpath, self.blackenvpath] + self.windowpath
        radmtx.rcvr_oct(rcvr_sun, cdsenv, sun_oct)
        vcdfmx = {}
        vcdrmx = {}
        vmap_paths = {}
        cdmap_paths = {}
        for view, sndr in self.sndr_vus.items():
            vmap_paths[view] = pjoin(self.mtxdir, f'vmap_{view}.hdr')
            cdmap_paths[view] = pjoin(self.mtxdir, f'cdmap_{view}.hdr')
            vdict = self.viewdicts[view]
            vdict.pop('c', None)
            vdict.pop('pj', None)
            view_str = radutil.opt2str(vdict)
            cmd = ['rpict'] + view_str.split() + ['-ps', '1',
                                                  '-ab', '0', '-av']
            cmd.extend(['.31831', '.31831', '.31831', self.vmap_oct])
            vmap = radutil.spcheckout(cmd)
            with open(vmap_paths[view], 'wb') as wtr:
                wtr.write(vmap)
            cmd[-1] = self.cdmap_oct
            cdmap = radutil.spcheckout(cmd)
            with open(cdmap_paths[view], 'wb') as wtr:
                wtr.write(cdmap)
            vcdfmx[view] = pjoin(self.mtxdir, f'vcdfmx_{view}')
            vcdrmx[view] = pjoin(self.mtxdir, f'vcdrmx_{view}')
            if not isdir(vcdrmx[view]) or self.config.overwrite:
                self.logger.info(f"Using rcontrib to generate direct sun r matrix for {view}...")
                radmtx.rcontrib(sender=sndr, modifier=rcvr_sun.modifier,
                                octree=sun_oct, out=vcdrmx[view],
                                opt=self.config.cdsmx_opt+' -i')
                _files = [pjoin(vcdrmx[view], f) for f in sorted(os.listdir(vcdrmx[view]))
                          if f.endswith('.hdr')]
                for idx, val in enumerate(_files):
                    os.rename(val, pjoin(vcdrmx[view], mod_names[idx]+'.hdr'))
            if not isdir(vcdfmx[view]) or self.config.overwrite:
                self.logger.info(f"Using rcontrib to generat direct sun f matrix for {view}...")
                radmtx.rcontrib(sender=sndr, modifier=rcvr_sun.modifier,
                                octree=sun_oct, out=vcdfmx[view],
                                opt=self.config.cdsmx_opt)
                _files = [pjoin(vcdfmx[view], f) for f in sorted(os.listdir(vcdfmx[view]))
                          if f.endswith('.hdr')]
                for idx, val in enumerate(_files):
                    os.rename(val, pjoin(vcdfmx[view], mod_names[idx]+'.hdr'))
        return vcdfmx, vcdrmx, vmap_paths, cdmap_paths

    def calc_5phase_pt(self, vmx, vmxd, dmx, dmxd, pcdsmx, smx, smxd, smx_sun):
        """."""
        self.logger.info(f"Computing sensor grid results")
        presl = []
        pdresl = []
        mult_cds = mtxmult(pcdsmx, smx_sun)
        prescd = [list(map(float, l.decode().strip().split('\t')))
                  for l in mult_cds.splitlines()]
        for wname in self.window_prims:
            _res = mtxmult(vmx[wname], self.bsdf[wname], dmx[wname], smx)
            _resd = mtxmult(vmxd[wname], self.bsdf[wname], dmxd[wname], smxd)
            presl.append([map(float, l.decode().strip().split('\t'))
                          for l in _res.splitlines()])
            pdresl.append([map(float, l.decode().strip().split('\t'))
                           for l in _resd.splitlines()])
        pres3 = [[sum(tup) for tup in zip(*line)] for line in zip(*presl)]
        pres3d = [[sum(tup) for tup in zip(*line)] for line in zip(*pdresl)]
        res = [[x-y+z for x, y, z in zip(a, b, c)]
               for a, b, c in zip(pres3, pres3d, prescd)]
        respath = pjoin(self.resdir, 'points5ph.txt')
        with open(respath, 'w') as wtr:
            for idx in range(len(res)):
                wtr.write(self.datetime_stamps[idx] + ',')
                wtr.write(','.join(map(str, res[idx])) + '\n')

    def calc_5phase_vu(self, vmx, vmxd, dmx, dmxd, vcdrmx, vcdfmx,
                       vmap_paths, cdmap_paths, smx, smxd, smx_sun):
        """Compute for image-based 5-phase method result."""
        for view in self.sndr_vus:
            self.logger.info(f"Computing for image-based results for {view}")
            vresl = []
            vdresl = []
            with tf.TemporaryDirectory() as td:
                vrescdr = tf.mkdtemp(dir=td)
                vrescdf = tf.mkdtemp(dir=td)
                vrescd = tf.mkdtemp(dir=td)
                cmds = []
                cmds.append(imgmult(pjoin(vcdrmx[view], '%04d.hdr'),
                                    smx_sun, odir=vrescdr))
                cmds.append(imgmult(pjoin(vcdfmx[view], '%04d.hdr'),
                                    smx_sun, odir=vrescdf))
                for wname in self.window_prims:
                    _vrespath = tf.mkdtemp(dir=td)
                    _vdrespath = tf.mkdtemp(dir=td)
                    cmds.append(imgmult(vmx[view+wname], self.bsdf[wname],
                                        dmx[wname], smx, odir=_vrespath))
                    cmds.append(imgmult(vmxd[view+wname], self.bsdf[wname],
                                        dmxd[wname], smxd, odir=_vdrespath))
                    vresl.append(_vrespath)
                    vdresl.append(_vdrespath)
                process = mp.Pool(int(self.config.nprocess))
                process.map(radutil.sprun, cmds)
                res3 = tf.mkdtemp(dir=td)
                res3di = tf.mkdtemp(dir=td)
                res3d = tf.mkdtemp(dir=td)
                if len(self.window_prims) > 1:
                    [vresl.insert(i*2-1, '+') for i in range(1, len(vresl))]
                    [vdresl.insert(i*2-1, '+') for i in range(1, len(vdresl))]
                    radutil.pcombop(vresl, res3)
                    radutil.pcombop(vdresl, res3di)
                else:
                    os.rename(vresl[0], res3)
                    os.rename(vdresl[0], res3di)
                radutil.pcombop([res3di, '*', vmap_paths[view]], res3d)
                radutil.pcombop(
                    [vrescdr, '*', cdmap_paths[view], '+', vrescdf], vrescd)
                opath = pjoin(self.resdir, f"{view}_5ph")
                if os.path.isdir(opath):
                    shutil.rmtree(opath)
                radutil.pcombop([res3, '-', res3d, '+', vrescd], opath)
                ofiles = [pjoin(opath, f) for f in sorted(os.listdir(opath)) if
                          f.endswith('.hdr')]
                [os.rename(ofiles[idx], pjoin(opath, self.datetime_stamps[idx]+'.hdr'))
                 for idx in range(len(ofiles))]
                self.logger.info(f"Done computing for {view}")
