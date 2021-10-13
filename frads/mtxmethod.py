"""
Typical Radiance matrix-based simulation workflows

TODO:
    1. all_material and blackenvpath causing rerunning hang, maybe move them to tmp
"""

import logging
import multiprocessing as mp
import os
import shutil
import subprocess as sp
import tempfile as tf
from typing import NamedTuple, List, Dict, Tuple
from frads import radutil, radgeom, radmtx, makesky, mfacade, mtxmult, util


logger = logging.getLogger('frads.mtxmethod')


class Model(NamedTuple):
    material_path: str
    window_groups: Dict[str, List[radutil.Primitive]]
    window_normals: List[radgeom.Vector]
    sender_grid: dict
    sender_view: dict
    views: dict
    receiver_sky: radmtx.Receiver
    cfs_paths: list
    blackenvpath: str


def get_wea(config, window_normals=None):
    """Obtain and prepare weather file data."""
    if config.wea_path != '':
        logger.info('Using user specified .wea file.')
        wea_path = os.path.join(config.rsodir, config.wea_path)
        with open(wea_path) as rdr:
            raw = rdr.read()
        sec = raw.split('\n\n')
        lines = [line.split() for line in sec[1].splitlines()]
        datetime_stamps = [
            "%02d%02d_%02d%02d" % (int(line[0]), int(line[1]),
                                   int(float(line[2])), 60*float(line[2]) % 1)
            for line in lines]
    else:
        if config.zipcode != '':
            logger.info('Downloading EPW file using zipcode.')
            lat, lon = util.get_latlon_from_zipcode(config.zipcode)
            epw_fname, url = util.get_epw_url(lat, lon)
        elif '' not in (config.latitude, config.longitude):
            logger.info('Downloading EPW file using lat&lon.')
            epw_fname, url = util.get_epw_url(
                float(config.latitude), float(config.longitude))
        else:
            raise NameError("Not site info defined")
        epw_str = util.request(url, {})
        logger.info("Downloaded: %s", epw_fname)
        logger.info('Converting EPW to a .wea file')
        if window_normals is None:
            remove_zero = False
            wea_name = util.basename(epw_fname) + '.wea'
        else:
            remove_zero = True
            wea_name = util.basename(epw_fname) + '_d6.wea'
        wea_metadata, wea_data = makesky.parse_epw(epw_str)
        if (config.start_hour != '') and (config.end_hour != ''):
            wea_data = makesky.start_end_hour(
                wea_data, config.start_hour, config.end_hour)
        if config.daylight_hours_only:
            wea_data = makesky.check_sun_above_horizon(wea_data, wea_metadata)
        if remove_zero:
            wea_data = makesky.remove_wea_zero_entry(
                wea_data, wea_metadata, window_normals)
        wea_path = os.path.join(config.rsodir, wea_name)
        datetime_stamps = []
        data_str = []
        for row in list(wea_data):
            datetime_stamps.append(row.dt_string())
            data_str.append(str(row))
        if data_str == []:
            raise ValueError("Empty wea file")
        with open(wea_path, 'w') as wtr:
            wtr.write(wea_metadata.wea_header())
            wtr.write('\n'.join(data_str))
    return wea_path, datetime_stamps


def gen_smx(wea_path, mfactor, outdir, onesun=False, direct=False):
    """Generate sky/sun matrix."""
    sun_only = ' -d' if direct else ''
    _five = ' -5 .533' if onesun else ''
    oname = util.basename(wea_path)
    cmd = f"gendaymtx -of -m {mfactor[-1]}{sun_only}{_five}".split()
    cmd.append(wea_path)
    logger.info('Generating sku/sun matrix using command')
    res = util.spcheckout(cmd)
    if direct:
        if onesun:
            smxpath = os.path.join(outdir, oname+'_d6.smx')
        else:
            smxpath = os.path.join(outdir, oname+'_d.smx')
    else:
        smxpath = os.path.join(outdir, oname+'.smx')
    with open(smxpath, 'wb') as wtr:
        wtr.write(res)
    return smxpath


def get_window_group(config: util.MradConfig) -> Tuple[dict, list]:
    window_groups = {}
    window_normals: List[radgeom.Vector] = []
    for wname, windowpath in config.windows.items():
        _window_primitives = radutil.unpack_primitives(windowpath)
        window_groups[wname] = _window_primitives
        _normal = radutil.parse_polygon(
            _window_primitives[0].real_arg).normal()
        window_normals.append(_normal)
    return window_groups, window_normals


def get_sender_grid(config: util.MradConfig) -> dict:
    """."""
    sender_grid = {}
    for name, surface_path in config.grid_surface_paths.items():
        surface_polygon = radutil.parse_polygon(
            radutil.unpack_primitives(surface_path)[0].real_arg)
        sensor_pts = radutil.gen_grid(
            surface_polygon,
            float(config.grid_height),
            float(config.grid_spacing))
        sender_grid[name] = radmtx.Sender.as_pts(
            pts_list=sensor_pts, ray_cnt=int(config.ray_count))
    return sender_grid


def get_sender_view(config: util.MradConfig) -> Tuple[dict, dict]:
    """Get views as senders.
    Args:
        config: MradConfig object"""
    sender_view = {}
    view_dicts = {}
    views = config.view.split(',') if config.view != '' else []
    for idx, view in enumerate(views):
        vdict = util.parse_vu(view)
        view_name = f"view_{idx:02d}"
        if 'vf' in vdict:
            with open(vdict['vf']) as rdr:
                vdict.update(util.parse_vu(rdr.read()))
        view_dicts[view_name] = vdict
        sender_view[view_name] = radmtx.Sender.as_view(
            vu_dict=vdict, ray_cnt=int(config.ray_count),
            xres=vdict['x'], yres=vdict['y'])
    return sender_view, view_dicts

def assemble_model(config: util.MradConfig) -> Model:
    """Assemble all the pieces together."""
    material_primitives: List[radutil.Primitive]
    material_primitives = sum([radutil.unpack_primitives(path) for path in config.material_paths], [])
    window_groups, _window_normals = get_window_group(config)
    window_normals = [item for idx, item in enumerate(_window_normals)
                      if item not in _window_normals[:idx]]
    sender_grid = get_sender_grid(config)
    sender_view, view_dicts = get_sender_view(config)
    rcvr_sky = radmtx.Receiver.as_sky(basis=config.smx_basis)
    # Sun CFS
    _cfs_path = []
    for wname, path in config.sun_cfs.items():
        ident = util.basename(path)
        if path.endswith('.xml'):
            window_primitives = window_groups[wname]
            upvec = radgeom.Vector(0, 0, 1)
            bsdf_prim = radutil.bsdf_prim('void', ident, path, upvec, pe=True)
            if bsdf_prim not in material_primitives:
                material_primitives.append(bsdf_prim)
            _tmp_cfs_path = f'tmpcfs{wname}.rad'
            with open(_tmp_cfs_path, 'w') as wtr:
                for primitive in window_primitives:
                    new_primitive = radutil.Primitive(
                        ident, primitive.ptype, primitive.identifier,
                        primitive.str_arg, primitive.real_arg)
                    wtr.write(str(new_primitive)+'\n')
            _cfs_path.append(_tmp_cfs_path)
        elif path.endswith('.rad'):
            _cfs_path.append(path)
    black_mat = radutil.Primitive('void', 'plastic', 'black', '0', '5 0 0 0 0 0')
    glow_mat = radutil.Primitive('void', 'glow', 'glowing', '0', '4 1 1 1 0')
    if black_mat not in material_primitives:
        material_primitives.append(black_mat)
    if glow_mat not in material_primitives:
        material_primitives.append(glow_mat)
    material_path = os.path.join(config.objdir, "all_material.mat")
    with open(material_path, 'w') as wtr:
        [wtr.write(str(primitive)+'\n') for primitive in material_primitives]
    _blackenvpath = os.path.join(config.objdir, 'blackened.rad')
    with open(_blackenvpath, 'w') as wtr:
        for path in config.scene_paths:
            wtr.write(f'\n!xform -m black {path}')
    return Model(material_path, window_groups, window_normals,
                 sender_grid, sender_view, view_dicts, rcvr_sky,
                 _cfs_path, _blackenvpath)


# def mtxmult(*mtx):
#     """Multiply matrices using rmtxop, convert to photopic, remove header."""
#     cmd1 = ['dctimestep', '-od'] + list(mtx)
#     cmd2 = ['rmtxop', '-fa', '-c', '47.4', '119.9', '11.6', '-', '-t']
#     cmd3 = ['getinfo', '-']
#     out1 = sp.Popen(cmd1, stdout=sp.PIPE)
#     out2 = sp.Popen(cmd2, stdin=out1.stdout, stdout=sp.PIPE)
#     out1.stdout.close()
#     out3 = sp.Popen(cmd3, stdin=out2.stdout, stdout=sp.PIPE)
#     out2.stdout.close()
#     out = out3.communicate()[0]
#     return out
#
#
# def imgmult(*mtx, odir):
#     """Image-based matrix multiplication using dctimestep."""
#     util.mkdir_p(odir)
#     cmd = ['dctimestep', '-oc', '-o', os.path.join(odir, '%04d.hdr')]
#     cmd += list(mtx)
#     return cmd


def prep_2phase_pt(model, config):
    """Prepare matrices two phase methods."""
    logger.info('Computing for 2-phase sensor point matrices...')
    env = [model.material_path] + config.scene_paths
    env += [v for v in config.windows.values()]
    pdsmx = os.path.join(config.mtxdir, 'pdsmx.mtx')
    opt = config.dsmx_opt + ' -n %s' % config.nprocess
    pdsmx = {}
    for grid_name, sender_grid in model.sender_grid.items():
        pdsmx[grid_name] = os.path.join(config.mtxdir, f"pdsmx_{grid_name}")
        if not os.path.isfile(pdsmx[grid_name]) or config.overwrite:
            res = radmtx.rfluxmtx(sender=sender_grid,
                                  receiver=model.receiver_sky,
                                  env=env, opt=opt)
            with open(pdsmx[grid_name], 'wb') as wtr:
                wtr.write(res)
    return pdsmx


def prep_2phase_vu(model, config):
    """Generate image-based matrices if view defined."""
    logger.info("Computing for image-based 2-phase matrices...")
    env = [model.material_path] + config.scene_paths
    env += [v for v in config.windows.values()]
    vdsmx = {}
    for view_name, sender_view in model.sender_view.items():
        vdsmx[view_name] = os.path.join(config.mtxdir, f"vdsmx_{view_name}")
        if not os.path.isdir(vdsmx[view_name]) or config.overwrite:
            logger.info("Generating for %s" % view_name)
            radmtx.rfluxmtx(sender=sender_view, receiver=model.receiver_sky,
                            env=env, opt=config.dsmx_opt, out=vdsmx[view_name])
    return vdsmx


def view_matrix_pt(model, config, direct=False):
    """."""
    _opt = config.vmx_opt
    _env = [model.material_path] + config.scene_paths
    if direct:
        logger.info("Computing direct view matrix for sensor grid:")
        _opt += ' -ab 1'
        _env = [model.material_path, model.blackenvpath]
    else:
        logger.info("Computing view matrix for sensor grid:")
    vmxs = {}
    receiver_windows = radmtx.Receiver(
        receiver='', basis=config.vmx_basis, modifier=None)
    for grid_name, sender_grid in model.sender_grid.items():
        for wname, window_prim in model.window_groups.items():
            _name = grid_name + wname
            if direct:
                _name += '_d'
            vmxs[grid_name + wname] = os.path.join(
                config.mtxdir, f'pvmx_{_name}.mtx')
            receiver_windows += radmtx.Receiver.as_surface(
                prim_list=window_prim, basis=config.vmx_basis,
                offset=None, left=None, source='glow',
                out=vmxs[grid_name + wname])
        files_exist = all([os.path.isfile(f) for f in vmxs.values()])
        if not files_exist or config.overwrite:
            logger.info("Generating vmx for %s", grid_name)
            radmtx.rfluxmtx(sender=sender_grid, receiver=receiver_windows,
                            env=_env, opt=_opt, out=None)
    return vmxs


def view_matrix_vu(model, config, direct=False):
    """Prepare matrices using three-phase methods."""
    _opt = config.vmx_opt
    _env = [model.material_path] + config.scene_paths
    if direct:
        _opt += ' -i -ab 1'
        _env = [model.material_path, model.blackenvpath]
    vmxs = {}
    vrcvr_windows = {}
    for view in model.sender_view:
        for wname in model.window_groups:
            vrcvr_windows[view+wname] = radmtx.Receiver(
                receiver='', basis=config.vmx_basis, modifier=None)
    for view in model.sender_view:
        logger.info("Computing image-based view matrix:")
        for wname, window_prim in model.window_groups.items():
            _name = view + wname
            if direct:
                _name += '_d'
            vmxs[view+wname] = os.path.join(
                config.mtxdir, f'vvmx_{_name}', '%04d.hdr')
            vrcvr_windows[view+wname] += radmtx.Receiver.as_surface(
                prim_list=window_prim, basis=config.vmx_basis,
                out=vmxs[view+wname])
    for view, sender_view in model.sender_view.items():
        for wname in model.window_groups:
            if not os.path.isdir(vmxs[view+wname][:-8]) or config.overwrite:
                logger.info("Generating for %s to %s", view, wname)
                util.mkdir_p(os.path.dirname(vmxs[view+wname]))
                radmtx.rfluxmtx(sender=sender_view,
                                receiver=vrcvr_windows[view+wname],
                                env=_env, opt=_opt, out=None)
    return vmxs


def facade_matrix(model, config, direct=False):
    """Generate facade matrices.
    Args:
        model (namedtuple): model assembly
        config (namedtuple): model configuration
    Returns:
        facade matrices file path
    """

    logger.info("Computing facade matrix...")
    fmxs = {}
    _opt = config.dmx_opt
    _env = [model.material_path] + config.scene_paths
    if direct:
        _opt += ' -ab 0'
        _env = [model.material_path, model.blackenvpath]
    ncp_prims = {}
    for ncppath in config.ncppath:
        name = util.basename(ncppath)
        ncp_prims[name] = radutil.unpack_primitives(ncppath)
    all_ncp_prims = [prim for _, prim in ncp_prims.items()]
    all_window_prims = [prim for key, prim in model.window_groups.items()]
    port_prims = mfacade.genport(wpolys=all_window_prims, npolys=all_ncp_prims,
                                 depth=None, scale=None)
    port_rcvr = radmtx.Receiver.as_surface(
        prim_list=port_prims, basis=config.fmx_basis, out=None)
    for wname in model.window_groups:
        _name = wname + '_d' if direct else wname
        fmxs[wname] = os.path.join(config.mtxdir, f'fmx_{_name}.mtx')
        window_prim = model.window_groups[wname]
        sndr_window = radmtx.Sender.as_surface(
            prim_list=window_prim, basis=config.fmx_basis)
        if not os.path.isfile(fmxs[wname]) or config.overwrite:
            logger.info("Generating facade matrix for %s", _name)
            fmx_res = radmtx.rfluxmtx(sender=sndr_window, receiver=port_rcvr,
                                      env=_env, out=None, opt=_opt)
            with open(fmxs[wname], 'wb') as wtr:
                wtr.write(fmx_res)
    return fmxs


def daylight_matrix(sender_prims, model, config, direct=False):
    """Call rfluxmtx to generate daylight matrices for each sender surface."""
    logger.info("Computing daylight matrix...")
    dmxs = {}
    _opt = config.dmx_opt
    _env = [model.material_path] + config.scene_paths
    if direct:
        _opt += ' -ab 0'
        _env = [model.material_path, model.blackenvpath]
    for sname, surface_primitives in sender_prims.items():
        _name = sname
        if direct:
            _name += '_d'
        dmxs[sname] = os.path.join(config.mtxdir, f'dmx_{_name}.mtx')
        sndr_window = radmtx.Sender.as_surface(
            prim_list=surface_primitives, basis=config.vmx_basis)
        if not os.path.isfile(dmxs[sname]) or config.overwrite:
            logger.info("Generating daylight matrix for %s", _name)
            dmx_res = radmtx.rfluxmtx(sender=sndr_window,
                                      receiver=model.receiver_sky,
                                      env=_env, out=None, opt=_opt)
            with open(dmxs[sname], 'wb') as wtr:
                wtr.write(dmx_res)
    return dmxs


def blacken_env(model, config):
    """."""
    bwindow_path = os.path.join(config.objdir, 'blackened_window.rad')
    gwindow_path = os.path.join(config.objdir, 'glowing_window.rad')
    blackened_window = []
    glowing_window = []
    for _, windows in model.window_groups.items():
        for window in windows:
            blackened_window.append(
                radutil.Primitive('black', window.ptype, window.identifier,
                                  window.str_arg, window.real_arg))
            glowing_window.append(
                radutil.Primitive('glowing', window.ptype, window.identifier,
                                  window.str_arg, window.real_arg))
    with open(bwindow_path, 'w') as wtr:
        wtr.write('\n'.join(list(map(str, blackened_window))))
    with open(gwindow_path, 'w') as wtr:
        wtr.write('\n'.join(list(map(str, glowing_window))))
    vmap_oct = os.path.join(config.rsodir, 'vmap.oct')
    cdmap_oct = os.path.join(config.rsodir, 'cdmap.oct')
    vmap = util.spcheckout(['oconv', model.material_path]+config.scene_paths+[bwindow_path])
    with open(vmap_oct, 'wb') as wtr:
        wtr.write(vmap)
    cdmap = util.spcheckout(['oconv', model.material_path]+config.scene_paths+[gwindow_path])
    with open(cdmap_oct, 'wb') as wtr:
        wtr.write(cdmap)
    return vmap_oct, cdmap_oct


def calc_4phase_pt(model, vmx, fmx, dmx, smx, datetime_stamps, config):
    """."""
    logger.info("Computing for 3-phase sensor grid results")
    presl = []
    with tf.TemporaryDirectory() as td:
        fdmx_path = os.path.join(td, 'fdmx.mtx')
        fdmx_res = radutil.rmtxop(fmx, dmx)
        with open(fdmx_path, 'wb') as wtr:
            wtr.write(fdmx_res)
        for wname in model.window_groups:
            _res = mtxmult.mtxmult(vmx[wname], config.klems_bsdfs[wname], fmx[wname],
                           dmx[wname], smx).splitlines()
            presl.append([map(float, l.decode().strip().split('\t'))
                          for l in _res])
        res = [[sum(tup) for tup in zip(*line)]for line in zip(*presl)]
        respath = os.path.join(config.resdir, 'points3ph.txt')
        with open(respath, 'w') as wtr:
            for idx, val in enumerate(res):
                wtr.write(datetime_stamps[idx] + ',')
                wtr.write(','.join(map(str, val)) + '\n')


def prep_4phase_pt(model, config, direct=False):
    """Prepare matrices using four-phase methods for point-based calculation."""
    dmxs = {}
    fmxs = {}
    _opt = config.fmx_opt
    _env = config.envpath
    if direct:
        # need to add ncp path
        _env = [model.material_path, model.blackenvpath]
        _opt += ' -ab 0'
    ncp_prims = None
    for wname in model.window_groups:
        window_prim = model.window_groups[wname]
        if direct:
            _name = wname + '_d'
        fmxs[wname] = os.path.join(config.mtxdir, 'fmx_{_name}.mtx')
        port_prims = mfacade.genport(
            wpolys=window_prim, npolys=ncp_prims, depth=None, scale=None)
        mfacade.Genfmtx(win_polygons=window_prim['polygon'], port_prim=port_prims,
                        out=fmxs[wname], env=_env, sbasis=config.vmx_basis,
                        rbasis=config.fmx_basis, opt=_opt, refl=False,
                        forw=False, wrap=False)
        logger.info(f"Generating daylight matrix for {wname}")
        dmxs[wname] = os.path.join(config.mtxdir, f'dmx_{wname}.mtx')
        sndr_port = radmtx.Sender.as_surface(
            prim_list=port_prims, basis=config.fmx_basis, offset=None)
        radmtx.rfluxmtx(sender=sndr_port, receiver=model.receiver_sky,
                        env=_env, out=dmxs[wname], opt=config.dmx_opt)
    return fmxs, dmxs


def direct_sun_matrix_pt(model, smx_path, config):
    """Compute direct sun matrix for sensor points.
    Args:
        smx_path: path to sun only sky matrix
    Returns:
        path to resulting direct sun matrix
    """

    logger.info("Direct sun matrix for sensor grid")
    rcvr_sun = radmtx.Receiver.as_sun(
        basis='r6', smx_path=smx_path,
        window_normals=model.window_normals, full_mod=True)
    sun_oct = os.path.join(config.rsodir, 'sun.oct')
    cdsenv = [model.material_path, model.blackenvpath] + model.cfs_paths
    radmtx.rcvr_oct(rcvr_sun, cdsenv, sun_oct)
    pcdsmx = {}
    for grid_name, sender_grid in model.sender_grid.items():
        pcdsmx[grid_name] = os.path.join(
            config.mtxdir, f'pcdsmx_{grid_name}.mtx')
        if not os.path.isfile(pcdsmx[grid_name]) or config.overwrite:
            logger.info("Generating using rcontrib...")
            radmtx.rcontrib(sender=sender_grid, modifier=rcvr_sun.modifier,
                            octree=sun_oct, out=pcdsmx[grid_name],
                            opt=config.cdsmx_opt)
    return pcdsmx


def direct_sun_matrix_vu(model, smx_path, vmap_oct, cdmap_oct, config):
    """Compute direct sun matrix for images.
    Args:
        smx_path: path to sun only sky matrix
    Returns:
        path to resulting direct sun matrix
    """
    logger.info("Generating image-based direct sun matrix")
    rcvr_sun = radmtx.Receiver.as_sun(
        basis='r6', smx_path=smx_path, window_normals=model.window_normals)
    mod_names = ["%04d" % (int(line[3:])-1)
                 for line in rcvr_sun.modifier.splitlines()]
    sun_oct = os.path.join(config.rsodir, 'sun.oct')
    cdsenv = [model.material_path, model.blackenvpath] + model.cfs_paths
    radmtx.rcvr_oct(rcvr_sun, cdsenv, sun_oct)
    vcdfmx = {}
    vcdrmx = {}
    vmap_paths = {}
    cdmap_paths = {}
    for view, sndr in model.sender_view.items():
        vmap_paths[view] = os.path.join(config.mtxdir, f'vmap_{view}.hdr')
        cdmap_paths[view] = os.path.join(config.mtxdir, f'cdmap_{view}.hdr')
        vdict = model.views[view]
        vdict.pop('c', None)
        vdict.pop('pj', None)
        view_str = radutil.opt2str(vdict)
        cmd = ['rpict'] + view_str.split() + ['-ps', '1',
                                              '-ab', '0', '-av']
        cmd.extend(['.31831', '.31831', '.31831', vmap_oct])
        vmap = util.spcheckout(cmd)
        with open(vmap_paths[view], 'wb') as wtr:
            wtr.write(vmap)
        cmd[-1] = cdmap_oct
        cdmap = util.spcheckout(cmd)
        with open(cdmap_paths[view], 'wb') as wtr:
            wtr.write(cdmap)
        vcdfmx[view] = os.path.join(config.mtxdir, f'vcdfmx_{view}')
        vcdrmx[view] = os.path.join(config.mtxdir, f'vcdrmx_{view}')
        tempf = os.path.join(config.mtxdir, 'vcdfmx')
        tempr = os.path.join(config.mtxdir, 'vcdrmx')
        if not os.path.isdir(vcdfmx[view]) or config.overwrite:
            logger.info(
                f"Using rcontrib to generat direct sun f matrix for {view}...")
            radmtx.rcontrib(sender=sndr, modifier=rcvr_sun.modifier,
                            octree=sun_oct, out=tempf,
                            opt=config.cdsmx_opt + ' -n %s' % config.nprocess)
            _files = [os.path.join(tempf, f) for f in sorted(os.listdir(tempf))
                      if f.endswith('.hdr')]
            util.mkdir_p(vcdfmx[view])
            for idx, val in enumerate(_files):
                shutil.move(val, os.path.join(vcdfmx[view], mod_names[idx] + '.hdr'))
            shutil.rmtree(tempf)
        if not os.path.isdir(vcdrmx[view]) or config.overwrite:
            logger.info(
                f"Using rcontrib to generate direct sun r matrix for {view}")
            radmtx.rcontrib(sender=sndr, modifier=rcvr_sun.modifier,
                            octree=sun_oct, out=tempr,
                            opt=config.cdsmx_opt + ' -i -n %s' % config.nprocess)
            _files = [os.path.join(tempr, f) for f in sorted(os.listdir(tempr))
                      if f.endswith('.hdr')]
            util.mkdir_p(vcdrmx[view])
            for idx, val in enumerate(_files):
                shutil.move(val, os.path.join(vcdrmx[view], mod_names[idx] + '.hdr'))
            shutil.rmtree(tempr)
    return vcdfmx, vcdrmx, vmap_paths, cdmap_paths


def calc_2phase_pt(model, datetime_stamps, dsmx, smx, config):
    """."""
    logger.info("Computing for 2-phase sensor grid results.")
    for grid_name in dsmx:
        grid_lines = model.sender_grid[grid_name].sender.decode().strip().splitlines()
        xypos = [",".join(line.split()[:3]) for line in grid_lines]
        opath = os.path.join(
            config.resdir, f'grid_{config.name}_{grid_name}.txt')
        res = mtxmult.mtxmult(dsmx[grid_name], smx)
        if isinstance(res, bytes):
            res = res.decode().splitlines()
        else:
            res = ["\t".join(map(str, row)) for row in res.T.tolist()]
        with open(opath, 'w') as wtr:
            wtr.write("\t"+"\t".join(xypos)+"\n")
            for idx, _ in enumerate(res):
                wtr.write(datetime_stamps[idx] + '\t')
                wtr.write(res[idx].rstrip() + '\n')


def calc_2phase_vu(datetime_stamps, dsmx, smx, config):
    """."""
    logger.info("Computing for 2-phase image-based results")
    for view in dsmx:
        opath = os.path.join(config.resdir, f'view_{config.name}_{view}')
        if os.path.isdir(opath):
            shutil.rmtree(opath)
        util.sprun(
            mtxmult.imgmult(os.path.join(dsmx[view], '%04d.hdr'), smx, odir=opath))
        ofiles = [os.path.join(opath, f) for f in sorted(os.listdir(opath))
                  if f.endswith('.hdr')]
        for idx, val in enumerate(ofiles):
            shutil.move(val, os.path.join(opath, datetime_stamps[idx]+'.hdr'))


def calc_3phase_pt(model, datetime_stamps, vmx, dmx, smx, config):
    """."""
    logger.info("Computing for 3-phase sensor grid results")
    for grid_name in model.sender_grid:
        presl = []
        for wname in model.window_groups:
            _res = mtxmult.mtxmult(vmx[grid_name+wname], config.klems_bsdfs[wname],
                           dmx[wname], smx)
            if isinstance(_res, bytes):
                _res = _res.splitlines()
                presl.append([map(float, line.decode().strip().split('\t'))
                              for line in _res])
            else:
                presl.append(_res.T.tolist())
        res = [[sum(tup) for tup in zip(*line)] for line in zip(*presl)]
        respath = os.path.join(config.resdir, f'grid_{config.name}_{grid_name}.txt')
        with open(respath, 'w') as wtr:
            for idx, val in enumerate(res):
                wtr.write(datetime_stamps[idx] + ',')
                wtr.write(','.join(map(str, val)) + '\n')


def calc_3phase_vu(model, datetime_stamps, vmx, dmx, smx, config):
    """."""
    logger.info("Computing for 3-phase image-based results:")
    for view in model.sender_view:
        opath = os.path.join(config.resdir, f'view_{config.name}_{view}')
        if os.path.isdir(opath):
            shutil.rmtree(opath)
        logger.info("for %s", view)
        vresl = []
        for wname in model.window_groups:
            _vrespath = os.path.join(config.resdir, f'{view}_{wname}')
            util.mkdir_p(_vrespath)
            cmd = mtxmult.imgmult(vmx[view+wname], config.klems_bsdfs[wname],
                          dmx[wname], smx, odir=_vrespath)
            util.sprun(cmd)
            vresl.append(_vrespath)
        if len(vresl) > 1:
            for i in range(1, len(vresl)):
                vresl.insert(i*2-1, '+')
            mtxmult.pcombop(vresl, opath)
            for path in vresl:
                if path != '+':
                    shutil.rmtree(path)
        else:
            shutil.move(vresl[0], opath)
        ofiles = [os.path.join(opath, f)
                  for f in sorted(os.listdir(opath))
                  if f.endswith('.hdr')]
        for idx, ofile in enumerate(ofiles):
            shutil.move(ofile, os.path.join(
                opath, datetime_stamps[idx]+'.hdr'))


def calc_5phase_pt(model, vmx, vmxd, dmx, dmxd, pcdsmx,
                   smx, smxd, smx_sun, datetime_stamps, config):
    """."""
    logger.info(f"Computing sensor grid results")
    for grid_name in model.sender_grid:
        presl = []
        pdresl = []
        mult_cds = mtxmult.mtxmult(pcdsmx[grid_name], smx_sun)
        if isinstance(mult_cds, bytes):
            prescd = [list(map(float, l.decode().strip().split('\t')))
                      for l in mult_cds.splitlines()]
        else:
            prescd = mult_cds.T.tolist()
        for wname in model.window_groups:
            _res = mtxmult.mtxmult(vmx[grid_name+wname], config.klems_bsdfs[wname], dmx[wname], smx)
            _resd = mtxmult.mtxmult(
                vmxd[grid_name+wname], config.klems_bsdfs[wname], dmxd[wname], smxd)
            if isinstance(_res, bytes):
                _res = [map(float, l.decode().strip().split('\t'))
                        for l in _res.splitlines()]
                _resd = [map(float, l.decode().strip().split('\t'))
                         for l in _resd.splitlines()]
            else:
                _res = _res.T.tolist()
                _resd = _resd.T.tolist()
            presl.append(_res)
            pdresl.append(_resd)
        pres3 = [[sum(tup) for tup in zip(*line)] for line in zip(*presl)]
        pres3d = [[sum(tup) for tup in zip(*line)] for line in zip(*pdresl)]
        res = [[x-y+z for x, y, z in zip(a, b, c)]
               for a, b, c in zip(pres3, pres3d, prescd)]
        respath = os.path.join(config.resdir, f'grid_{config.name}_{grid_name}.txt')
        with open(respath, 'w') as wtr:
            for idx in range(len(res)):
                wtr.write(datetime_stamps[idx] + ',')
                wtr.write(','.join(map(str, res[idx])) + '\n')


def calc_5phase_vu(model, vmx, vmxd, dmx, dmxd, vcdrmx, vcdfmx,
                   vmap_paths, cdmap_paths, smx, smxd, smx_sun,
                   datetime_stamps, datetime_stamps_d6, config):
    """Compute for image-based 5-phase method result."""
    for view in model.sender_view:
        logger.info(f"Computing for image-based results for {view}")
        vresl = []
        vdresl = []
        with tf.TemporaryDirectory() as td:
            vrescdr = tf.mkdtemp(dir=td)
            vrescdf = tf.mkdtemp(dir=td)
            vrescd = tf.mkdtemp(dir=td)
            cmds = []
            cmds.append(mtxmult.imgmult(os.path.join(vcdrmx[view], '%04d.hdr'),
                                smx_sun, odir=vrescdr))
            cmds.append(mtxmult.imgmult(os.path.join(vcdfmx[view], '%04d.hdr'),
                                smx_sun, odir=vrescdf))
            for wname in model.window_groups:
                _vrespath = tf.mkdtemp(dir=td)
                _vdrespath = tf.mkdtemp(dir=td)
                cmds.append(mtxmult.imgmult(vmx[view+wname], config.klems_bsdfs[wname],
                                    dmx[wname], smx, odir=_vrespath))
                cmds.append(mtxmult.imgmult(vmxd[view+wname], config.klems_bsdfs[wname],
                                    dmxd[wname], smxd, odir=_vdrespath))
                vresl.append(_vrespath)
                vdresl.append(_vdrespath)
            logger.info("Multiplying matrices for images.")
            process = mp.Pool(int(config.nprocess))
            process.map(util.sprun, cmds)
            res3 = tf.mkdtemp(dir=td)
            res3di = tf.mkdtemp(dir=td)
            res3d = tf.mkdtemp(dir=td)
            logger.info("Combine results for each window groups.")
            if len(model.window_groups) > 1:
                [vresl.insert(i*2-1, '+') for i in range(1, len(vresl))]
                [vdresl.insert(i*2-1, '+') for i in range(1, len(vdresl))]
                mtxmult.pcombop(
                    vresl, res3, nproc=int(config.nprocess))
                mtxmult.pcombop(
                    vdresl, res3di, nproc=int(config.nprocess))
            else:
                shutil.move(vresl[0], res3)
                shutil.move(vdresl[0], res3di)
            logger.info("Applying mapterial reflectance map")
            mtxmult.pcombop([res3di, '*', vmap_paths[view]],
                            res3d, nproc=int(config.nprocess))
            mtxmult.pcombop([vrescdr, '*', cdmap_paths[view], '+', vrescdf],
                            vrescd, nproc=int(config.nprocess))
            opath = os.path.join(config.resdir, f"view_{config.name}_{view}")
            if os.path.isdir(opath):
                shutil.rmtree(opath)
            logger.info("Assemble all phase results.")
            res3_path = [os.path.join(res3, f) for f in sorted(
                os.listdir(res3)) if f.endswith('.hdr')]
            [shutil.move(file, os.path.join(res3, datetime_stamps[idx]+'.hdr'))
             for idx, file in enumerate(res3_path)]
            res3d_path = [os.path.join(res3d, f) for f in sorted(
                os.listdir(res3d)) if f.endswith('.hdr')]
            [shutil.move(file, os.path.join(res3d, datetime_stamps[idx]+'.hdr'))
             for idx, file in enumerate(res3d_path)]
            vrescd_path = [os.path.join(vrescd, f) for f in sorted(
                os.listdir(vrescd)) if f.endswith('.hdr')]
            [shutil.move(file, os.path.join(vrescd, datetime_stamps_d6[idx]+'.hdr'))
             for idx, file in enumerate(vrescd_path)]
            util.mkdir_p(opath)
            for hdr3 in os.listdir(res3):
                _hdr3 = os.path.join(res3, hdr3)
                _hdr3d = os.path.join(res3d, hdr3)
                cmd = f"pcomb -o {_hdr3} -s -1 -o {_hdr3d}"
                if hdr3 in os.listdir(vrescd):
                    cmd += ' -o ' + os.path.join(vrescd, hdr3)
                process = sp.run(cmd.split(), stdout=sp.PIPE)
                with open(os.path.join(opath, hdr3), 'wb') as wtr:
                    wtr.write(process.stdout)
            logger.info(f"Done computing for {view}")


def two_phase(model, config):
    """Two-phase simulation workflow."""
    wea_path, datetime_stamps = get_wea(config)
    smx = gen_smx(wea_path, config.smx_basis, config.mtxdir)
    pdsmx = prep_2phase_pt(model, config)
    vdsmx = prep_2phase_vu(model, config)
    if not config.no_multiply:
        calc_2phase_pt(model, datetime_stamps, pdsmx, smx, config)
        calc_2phase_vu(datetime_stamps, vdsmx, smx, config)
    return pdsmx, vdsmx


def three_phase(model, config, direct=False):
    """3/5-phase simulation workflow."""
    wea_path, datetime_stamps = get_wea(config)
    smx = gen_smx(wea_path, config.smx_basis, config.mtxdir)
    pvmxs = view_matrix_pt(model, config)
    vvmxs = view_matrix_vu(model, config)
    dmxs = daylight_matrix(model.window_groups, model, config)
    if direct:
        wea_path_d6, datetime_stamps_d6 = get_wea(
            config, window_normals=model.window_normals)
        smxd = gen_smx(wea_path, config.smx_basis, config.mtxdir, direct=True)
        smx_sun_img = gen_smx(wea_path_d6, config.cdsmx_basis,
                              config.mtxdir, onesun=True, direct=True)
        smx_sun = gen_smx(wea_path, config.cdsmx_basis, config.mtxdir,
                          onesun=True, direct=True)
        vmap_oct, cdmap_oct = blacken_env(model, config)
        pcdsmx = direct_sun_matrix_pt(model, smx_sun, config)
        vcdfmx, vcdrmx, vmap_paths, cdmap_paths = direct_sun_matrix_vu(
            model, smx_sun_img, vmap_oct, cdmap_oct, config)
        dmxsd = daylight_matrix(model.window_groups,
                                model, config, direct=True)
        pvmxsd = view_matrix_pt(model, config, direct=True)
        vvmxsd = view_matrix_vu(model, config, direct=True)
        if not config.no_multiply:
            calc_5phase_pt(model, pvmxs, pvmxsd, dmxs, dmxsd, pcdsmx,
                           smx, smxd, smx_sun, datetime_stamps, config)
            calc_5phase_vu(model, vvmxs, vvmxsd, dmxs, dmxsd, vcdfmx,
                           vcdrmx, vmap_paths, cdmap_paths, smx, smxd,
                           smx_sun_img, datetime_stamps, datetime_stamps_d6, config)
    else:
        if not config.no_multiply:
            calc_3phase_pt(model, datetime_stamps, pvmxs, dmxs, smx, config)
            calc_3phase_vu(model, datetime_stamps, vvmxs, dmxs, smx, config)


def four_phase(model, config, direct=False):
    """Four-phase simulation workflow."""
    wea_path, datetime_stamps = get_wea(config)
    smx = gen_smx(wea_path, config.smx_basis, config.mtxdir)
    pvmxs = view_matrix_pt(model, config)
    vvmxs = view_matrix_vu(model, config)
    fmxs = facade_matrix(model, config)
    dmxs = daylight_matrix(model.port_prims, model, config)
    if direct:
        wea_path_d6, datetime_stamps_d6 = get_wea(
            config, window_normals=model.window_normals)
        smxd = gen_smx(wea_path, config.smx_basis, config.mtxdir, direct=True)
        smx_sun_img = gen_smx(wea_path_d6, config.cdsmx_basis,
                              config.mtxdir, onesun=True, direct=True)
        smx_sun = gen_smx(wea_path, config.cdsmx_basis, config.mtxdir,
                          onesun=True, direct=True)
        vmap_oct, cdmap_oct = blacken_env(model, config)
        pcdsmx = direct_sun_matrix_pt(model, smx_sun, config)
        vcdfmx, vcdrmx, vmap_paths, cdmap_paths = direct_sun_matrix_vu(
            model, smx_sun_img, vmap_oct, cdmap_oct, config)
        dmxsd = daylight_matrix(model.window_groups,
                                model, config, direct=True)
        pvmxsd = view_matrix_pt(model, config, direct=True)
        vvmxsd = view_matrix_vu(model, config, direct=True)
        calc_5phase_pt(model, pvmxs, pvmxsd, dmxs, dmxsd, pcdsmx,
                       smx, smxd, smx_sun, datetime_stamps, config)
        calc_5phase_vu(model, vvmxs, vvmxsd, dmxs, dmxsd, vcdfmx,
                       vcdrmx, vmap_paths, cdmap_paths, smx, smxd,
                       smx_sun_img, datetime_stamps, datetime_stamps_d6, config)
    else:
        calc_4phase_pt(pvmxs, fmxs, dmxs, smx)
        # calc_4phase_vu(vvmxs, fmxs, dmxs, smx)




#     def prep_4phase_vu(self):
#         """."""
#         vvmxs = {}
#         dmxs = {}
#         fmxs = {}
#         prcvr_windows = radmtx.Receiver(
#             receiver='', basis=self.vmx_basis, modifier=None)
#         if len(self.sndr_views) > 0:
#             vrcvr_windows = {}
#             for view in model.sender_view:
#                 vrcvr_windows[view] = radmtx.Receiver(
#                     receiver='', basis=self.vmx_basis, modifier=None)
#         port_prims = mfacade.genport
#             wpolys=window_prims, npolys=ncp_prims, depth=None, scale=None)
#         mfacade.Genfmtx(win_polygons=window_polygon, port_prim=port_prims,
#                         out=kwargs['o'], env=kwargs['env'], sbasis=kwargs['ss'],
#                         rbasis=kwargs['rs'], opt=kwargs['opt'], refl=False,
#                         forw=False, wrap=False)
#         for wname in model.window_groups:
#             window_prim = model.window_groups[wname]
#             logger.info(f"Generating daylight matrix for {wname}")
#             dmxs[wname] = os.path.join(self.mtxdir, f'dmx_{wname}.mtx')
#             sndr_window = radmtx.Sender.as_surface(
#                 prim_list=window_prim, basis=self.vmx_basis, offset=None)
#             sndr_port = radmtx.Sender.as_surface(
#                 prim_list=port_prims, basis=self.fmx_basis, offset=None)
#             radmtx.rfluxmtx(sender=sndr_port, receiver=model.receiver_sky,
#                             env=self.envpath, out=dmxs[wname], opt=self.dmx_opt)
#             for view in model.sender_view:
#                 vvmxs[view+wname] = os.path.join(
#                     self.mtxdir, f'vvmx_{view}_{wname}', '%04d.hdr')
#                 util.mkdir_p(os.path.dirname(vvmxs[view+wname]))
#                 vrcvr_windows[view] += radmtx.Receiver.as_surface(
#                     prim_list=window_prim, basis=self.vmx_basis,
#                     offset=None, left=None, source='glow', out=vvmxs[view+wname])
#         logger.info(f"Generating view matrix for {view}")
#         for view in model.sender_view:
#             radmtx.rfluxmtx(sender=model.sender_view[view], receiver=vrcvr_windows[view],
#                             env=self.envpath, opt=self.vmx_opt, out=None)

