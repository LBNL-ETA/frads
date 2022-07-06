"""
Typical Radiance matrix-based simulation workflows
"""

from configparser import ConfigParser
import logging
import os
from pathlib import Path
import shutil
import subprocess as sp
import tempfile as tf
from typing import List
from typing import Dict
from typing import Sequence
from typing import Tuple

from frads import geom
from frads import sky
from frads import matrix
# from frads import ncp
from frads import mtxmult
from frads import parsers
from frads import utils
from frads.types import Primitive
from frads.types import Receiver
from frads.types import MradModel
from frads.types import MradPath
from frads.types import WeaMetaData
from frads.types import WeaDataRow


logger = logging.getLogger("frads.methods")


def get_window_group(config: ConfigParser) -> Tuple[dict, list]:
    """Parse window groups from config."""
    window_groups = {}
    window_normals: List[geom.Vector] = []
    for wpath in config["Model"]["window_paths"].split():
        wname: str = Path(wpath).stem
        _window_primitives = utils.unpack_primitives(wpath)
        window_groups[wname] = _window_primitives
        _normal = parsers.parse_polygon(_window_primitives[0].real_arg).normal()
        window_normals.append(_normal)
    return window_groups, window_normals


def get_wea_data(config: ConfigParser) -> Tuple[WeaMetaData, List[WeaDataRow], str]:
    """Get wea data and parse into appropriate data types."""
    if (wea_path := config["Site"]["wea_path"]) != "":
        logger.info("Using user specified %s file." % wea_path)
        name = Path(wea_path).stem
        with open(wea_path) as rdr:
            wea_metadata, wea_data = parsers.parse_wea(rdr.read())
    elif (epw_path := config["Site"]["epw_path"]) != "":
        logger.info(f"Converting {epw_path} to a .wea file")
        name = Path(epw_path).stem
        with open(epw_path, "r") as rdr:
            wea_metadata, wea_data = parsers.parse_epw(rdr.read())
    else:
        raise ValueError("Need either a .wea or a .epw file")
    return wea_metadata, wea_data, name


def get_sender_grid(config: ConfigParser) -> Dict[str, matrix.Sender]:
    """Get point grid as ray senders."""
    sender_grid = {}
    for gpath in config["RaySender"]["grid_surface"].split():
        name: str = Path(gpath).stem
        # Take the first polygon primitive
        gprimitives = utils.unpack_primitives(gpath)
        surface_polygon = None
        for prim in gprimitives:
            if prim.ptype == "polygon":
                surface_polygon = parsers.parse_polygon(prim.real_arg)
        if surface_polygon is None:
            raise ValueError(f"No polygon found in {gpath}")
        sensor_pts = utils.gen_grid(
            surface_polygon,
            float(config["RaySender"]["grid_height"]),
            float(config["RaySender"]["grid_spacing"]),
        )
        sender_grid[name] = matrix.points_as_sender(
            pts_list=sensor_pts, ray_cnt=int(config["SimControl"]["ray_count"])
        )
    return sender_grid


def get_sender_view(config: ConfigParser):
    """Get a single view as a sender.
    Args:
        config: MradConfig object"""
    sender_view: Dict[str, matrix.Sender] = {}
    view_dicts: Dict[str, dict] = {}
    try:
        view_str = config["RaySender"]["view"]
    except KeyError:
        return sender_view, view_dicts
    vdict = parsers.parse_vu(view_str)
    view_name = "view_00"
    if "vf" in vdict:
        with open(vdict["vf"]) as rdr:
            vdict.update(parsers.parse_vu(rdr.read()))
    view_dicts[view_name] = vdict
    sender_view[view_name] = matrix.view_as_sender(
        vu_dict=vdict,
        ray_cnt=int(config["SimControl"]["ray_count"]),
        xres=vdict["x"],
        yres=vdict["y"],
    )
    return sender_view, view_dicts


def assemble_model(config: ConfigParser) -> MradModel:
    """Assemble all the pieces together."""
    material_primitives: List[Primitive] = []
    for path in config["Model"]["material"].split():
        material_primitives.extend(utils.unpack_primitives(path))
    window_groups, _window_normals = get_window_group(config)
    window_normals = [
        item
        for idx, item in enumerate(_window_normals)
        if item not in _window_normals[:idx]
    ]
    sender_grid = get_sender_grid(config)
    sender_view, view_dicts = get_sender_view(config)
    rcvr_sky = matrix.sky_as_receiver(basis=config["SimControl"]["smx_basis"])
    cfs_path = []
    if "window_cfs" in config["Model"]:
        cfs_path = config["Model"]["window_cfs"].split()
    bsdf_mat = {
        wname: Path(path)
        for wname, path in zip(window_groups, config["Model"]["window_xml"].split())
    }
    black_mat = Primitive("void", "plastic", "black", "0", "5 0 0 0 0 0")
    glow_mat = Primitive("void", "glow", "glowing", "0", "4 1 1 1 0")
    if black_mat not in material_primitives:
        material_primitives.append(black_mat)
    if glow_mat not in material_primitives:
        material_primitives.append(glow_mat)
    # _, material_path = tf.mkstemp(suffix="all_material")
    material_path = "all_material" + utils.id_generator()
    with open(material_path, "w") as wtr:
        [wtr.write(str(primitive) + "\n") for primitive in material_primitives]
    _blackenvpath = "blackened.rad"
    with open(_blackenvpath, "w") as wtr:
        for path in config["Model"]["scene"].split():
            wtr.write(f"\n!xform -m black {path}")
    return MradModel(
        material_path,
        window_groups,
        window_normals,
        sender_grid,
        sender_view,
        view_dicts,
        rcvr_sky,
        bsdf_mat,
        cfs_path,
        _blackenvpath,
    )


def prep_2phase_pt(mpath: MradPath, model: MradModel, config: ConfigParser) -> None:
    """Prepare matrices two phase methods."""
    logger.info("Computing for 2-phase sensor point matrices...")
    env = [model.material_path] + config["Model"]["scene"].split()
    env += config["Model"]["window_paths"].split()
    opt = config["SimControl"]["dsmx_opt"]
    opt += f' -n {config["SimControl"]["nprocess"]}'
    overwrite = config.getboolean("SimControl", "overwrite", fallback=False)
    for grid_name, sender_grid in model.sender_grid.items():
        mpath.pdsmx[grid_name] = Path("Matrices", f"pdsmx_{grid_name}.mtx")
        if (not mpath.pdsmx[grid_name].is_file()) or overwrite:
            res = matrix.rfluxmtx(
                sender=sender_grid, receiver=model.receiver_sky, env=env, opt=opt
            )
            with open(mpath.pdsmx[grid_name], "wb") as wtr:
                wtr.write(res)


def prep_2phase_vu(mpath: MradPath, model: MradModel, config: ConfigParser) -> None:
    """Generate image-based matrices if view defined."""
    logger.info("Computing for image-based 2-phase matrices...")
    env = [model.material_path] + config["Model"]["scene"].split()
    env += config["Model"]["window_paths"].split()
    opt = config["SimControl"]["dsmx_opt"]
    opt += f' -n {config["SimControl"]["nprocess"]}'
    overwrite = config.getboolean("SimControl", "overwrite", fallback=False)
    for view_name, sender_view in model.sender_view.items():
        mpath.vdsmx[view_name] = Path("Matrices", f"vdsmx_{view_name}")
        if (not mpath.vdsmx[view_name].is_dir()) or overwrite:
            logger.info("Generating for %s" % view_name)
            matrix.rfluxmtx(
                sender=sender_view,
                receiver=model.receiver_sky,
                env=env,
                opt=opt,
                out=mpath.vdsmx[view_name],
            )


def view_matrix_pt(
    mpath: MradPath, model: MradModel, config: ConfigParser, direct=False
) -> None:
    """."""
    _opt = config["SimControl"]["vmx_opt"]
    _env: List[str] = [model.material_path] + config["Model"]["scene"].split()
    if direct:
        logger.info("Computing direct view matrix for sensor grid:")
        _opt += " -ab 1"
        _env = [model.material_path, model.blackenvpath]
    else:
        logger.info("Computing view matrix for sensor grid:")
    receiver_windows = Receiver(receiver="", basis=config["SimControl"]["vmx_basis"])
    for grid_name, sender_grid in model.sender_grid.items():
        for wname, window_prim in model.window_groups.items():
            _name = grid_name + wname
            if direct:
                mpath.pvmxd[_name] = Path("Matrices", f"pvmx_{_name}_d.mtx")
                out = mpath.pvmxd[_name]
            else:
                mpath.pvmx[_name] = Path("Matrices", f"pvmx_{_name}.mtx")
                out = mpath.pvmx[_name]
            receiver_windows += matrix.surface_as_receiver(
                prim_list=window_prim,
                basis=config["SimControl"]["vmx_basis"],
                offset=None,
                left=None,
                source="glow",
                out=out,
            )
        if direct:
            files_exist = all([f.is_file() for f in mpath.pvmxd.values()])
        else:
            files_exist = all([f.is_file() for f in mpath.pvmx.values()])
        if (not files_exist) or config.getboolean(
            "SimControl", "overwrite", fallback=False
        ):
            logger.info("Generating vmx for %s", grid_name)
            matrix.rfluxmtx(
                sender=sender_grid,
                receiver=receiver_windows,
                env=_env,
                opt=_opt,
                out=None,
            )


def view_matrix_vu(
    mpath: MradPath, model: MradModel, config: ConfigParser, direct=False
):
    """Prepare matrices using three-phase methods."""
    _opt = config["SimControl"]["vmx_opt"]
    _env = [model.material_path] + config["Model"]["scene"].split()
    direct_msg = " "
    if direct:
        _opt += " -i -ab 1"
        _env = [model.material_path, model.blackenvpath]
        direct_msg = " direct-only matrix "
    _opt += f' -n {config["SimControl"]["nprocess"]}'
    overwrite = config.getboolean("SimControl", "overwrite", fallback=False)
    logger.info("Computing image-based view matrix:")
    for view, sender_view in model.sender_view.items():
        vrcvr_windows = Receiver(
            receiver="", basis=config["SimControl"]["vmx_basis"]
        )
        for wname, window_prim in model.window_groups.items():
            _name = view + wname
            if direct:
                mpath.vvmxd[_name] = Path("Matrices", f"vvmx_{_name}_d", "%04d.hdr")
                out = mpath.vvmxd[_name]
            else:
                mpath.vvmx[_name] = Path("Matrices", f"vvmx_{_name}", "%04d.hdr")
                out = mpath.vvmx[_name]
            out.parent.mkdir(exist_ok=True)
            vrcvr_windows += matrix.surface_as_receiver(
                prim_list=window_prim,
                basis=config["SimControl"]["vmx_basis"],
                out=out,
            )
        if direct:
            exists = all([any(d.iterdir()) for d in mpath.vvmxd[_name].parents[1].glob("vvmx*_d")])
        else:
            exists = all([any(d.iterdir()) for d in mpath.vvmx[_name].parents[1].glob("vvmx*[!_d]")])
        if (not exists) or config.getboolean("SimControl", "overwrite"):
            matrix.rfluxmtx(sender=sender_view, receiver=vrcvr_windows, env=_env, opt=_opt, out=None)


# def facade_matrix(mpath: MradPath, model: MradModel, config: ConfigParser, direct=False) -> None:
#     """Generate facade matrices.
#     Args:
#         model (namedtuple): model assembly
#         config (namedtuple): model configuration
#     Returns:
#         facade matrices file path
#     """
#
#     logger.info("Computing facade matrix...")
#     fmxs = {}
#     fmx_opt = config["SimControl"]["fmx_opt"]
#     fmx_opt += f' -n {config["SimControl"]["nprocess"]}'
#     _env = [model.material_path] + config["Model"]["scene"].split()
#     overwrite = config.getboolean("SimControl", "overwrite", fallback=False)
#     if direct:
#         fmx_opt += " -ab 0"
#         _env = [model.material_path, model.blackenvpath]
#     ncp_prims = {}
#     for ncppath in config["Model"]["ncppath"].split():
#         name: str = Path(ncppath).stem
#         ncp_prims[name] = utils.unpack_primitives(ncppath)
#     all_ncp_prims = [prim for _, prim in ncp_prims.items()]
#     all_window_prims = [prim for key, prim in model.window_groups.items()]
#     port_prims = ncp.gen_port_prims_from_window_ncp(all_window_prims, all_ncp_prims)
#     port_rcvr = matrix.surface_as_receiver(
#         prim_list=port_prims, basis=config["SimControl"]["fmx_basis"], out=None
#     )
#     for wname in model.window_groups:
#         _name = wname + "_d" if direct else wname
#         fmxs[wname] = Path("Matrices", f"fmx_{_name}.mtx")
#         window_prim = model.window_groups[wname]
#         sndr_window = matrix.Sender.as_surface(
#             prim_list=window_prim, basis=config["SimControl"]["fmx_basis"]
#         )
#         if (not fmxs[wname].is_file()) or overwrite:
#             logger.info("Generating facade matrix for %s", _name)
#             fmx_res = matrix.rfluxmtx(
#                 sender=sndr_window, receiver=port_rcvr, env=_env, out=None, opt=fmx_opt
#             )
#             with open(fmxs[wname], "wb") as wtr:
#                 wtr.write(fmx_res)
#     return fmxs


def daylight_matrix(mpath: MradPath, model: MradModel, config: ConfigParser, direct=False) -> None:
    """Call rfluxmtx to generate daylight matrices for each sender surface."""
    logger.info("Computing daylight matrix...")
    dmx_opt = config["SimControl"]["dmx_opt"]
    dmx_opt += f' -n {config["SimControl"]["nprocess"]}'
    dmx_env = [model.material_path] + config["Model"]["scene"].split()
    if direct:
        dmx_opt += " -ab 0"
        dmx_env = [model.material_path, model.blackenvpath]
    for sname, surface_primitives in model.window_groups.items():
        _name = sname
        if direct:
            mpath.dmxd[sname] = Path("Matrices", f"dmx_{_name}_d.mtx")
            out=mpath.dmxd[sname]
        else:
            mpath.dmx[sname] = Path("Matrices", f"dmx_{_name}.mtx")
            out=mpath.dmx[sname]
        sndr_window = matrix.surface_as_sender(
            prim_list=surface_primitives, basis=config["SimControl"]["vmx_basis"]
        )
        if regen(out, config):
            logger.info("Generating daylight matrix for %s", _name)
            dmx_res = matrix.rfluxmtx(
                sender=sndr_window,
                receiver=model.receiver_sky,
                env=dmx_env,
                out=None,
                opt=dmx_opt,
            )
            with open(out, "wb") as wtr:
                wtr.write(dmx_res)


def blacken_env(model, config: ConfigParser):
    """."""
    bwindow_path = "blackened_window.rad"
    gwindow_path = "glowing_window.rad"
    blackened_window = []
    glowing_window = []
    for _, windows in model.window_groups.items():
        for window in windows:
            blackened_window.append(
                Primitive(
                    "black",
                    window.ptype,
                    window.identifier,
                    window.str_arg,
                    window.real_arg,
                )
            )
            glowing_window.append(
                Primitive(
                    "glowing",
                    window.ptype,
                    window.identifier,
                    window.str_arg,
                    window.real_arg,
                )
            )
    with open(bwindow_path, "w") as wtr:
        wtr.write("\n".join(list(map(str, blackened_window))))
    with open(gwindow_path, "w") as wtr:
        wtr.write("\n".join(list(map(str, glowing_window))))
    vmap_oct = "vmap.oct"
    cdmap_oct = "cdmap.oct"
    vmap_cmd = ["oconv", "-f", model.material_path] + config["Model"]["scene"].split() + [bwindow_path]
    with open(vmap_oct, "wb") as wtr:
        sp.run(vmap_cmd, check=True, stdout=wtr)
    cdmap_cmd = ["oconv", "-f", model.material_path] + config["Model"]["scene"].split() + [gwindow_path]
    with open(cdmap_oct, "wb") as wtr:
        sp.run(cdmap_cmd, check=True, stdout=wtr)
    os.remove(bwindow_path)
    os.remove(gwindow_path)
    return vmap_oct, cdmap_oct


def calc_4phase_pt(mpath, model, datetime_stamps, vmx, fmx, dmx, smx, config: ConfigParser):
    """."""
    logger.info("Computing for 4-phase sensor grid results")
    presl = []
    with tf.TemporaryDirectory() as td:
        fdmx_path = Path(td, "fdmx.mtx")
        fdmx_res = mtxmult.mtxmult(fmx, dmx)
        with open(fdmx_path, "wb") as wtr:
            wtr.write(fdmx_res)
        for wname in model.window_groups:
            _res = mtxmult.mtxmult(
                vmx[wname], model.bsdf_xml[wname], fmx[wname], dmx[wname], smx
            )
            presl.append(
                [
                    map(float, line.decode().strip().split("\t"))
                    for line in _res.splitlines()
                ]
            )
        res = [[sum(tup) for tup in zip(*line)] for line in zip(*presl)]
        respath = Path("Results", "points3ph.txt")
        with open(respath, "w") as wtr:
            for idx, val in enumerate(res):
                wtr.write(datetime_stamps[idx] + ",")
                wtr.write(",".join(map(str, val)) + "\n")


# def prep_4phase_pt(mpath, model, config: ConfigParser, direct=False) -> None:
#     """Prepare matrices using four-phase methods for point-based calculation."""
#     dmxs = {}
#     fmxs = {}
#     _opt = config["SimControl"]["fmx_opt"]
#     _env = [model.material_path] + config["Model"]["scene"].split()
#     if direct:
#         # need to add ncp path
#         _env = [model.material_path, model.blackenvpath]
#         _opt += " -ab 0"
#     ncp_prims = None
#     for wname in model.window_groups:
#         window_prim = model.window_groups[wname]
#         _name = wname
#         if direct:
#             _name += "_d"
#         fmxs[wname] = Path("Matrices", "fmx_{_name}.mtx")
#         port_prims = ncp.gen_ports_from_window_ncp(window_prim, ncp_prims)
#         ncp.gen_ncp_mtx(
#             win_polygons=window_prim["polygon"],
#             port_prim=port_prims,
#             out=fmxs[wname],
#             env=_env,
#             sbasis=config["SimControl"]["vmx_basis"],
#             rbasis=config["SimControl"]["fmx_basis"],
#             opt=_opt,
#             refl=False,
#             forw=False,
#             wrap=False,
#         )
#         logger.info(f"Generating daylight matrix for {wname}")
#         dmxs[wname] = Path("Matrices", f"dmx_{wname}.mtx")
#         sndr_port = matrix.Sender.as_surface(
#             prim_list=port_prims, basis=config["SimControl"]["fmx_basis"], offset=None
#         )
#         matrix.rfluxmtx(
#             sender=sndr_port,
#             receiver=model.receiver_sky,
#             env=_env,
#             out=dmxs[wname],
#             opt=config["SimControl"]["dmx_opt"],
#         )


def direct_sun_matrix_pt(mpath: MradPath, model: MradModel, config: ConfigParser) -> None:
    """Compute direct sun matrix for sensor points.
    Args:
        smx_path: path to sun only sky matrix
    Returns:
        path to resulting direct sun matrix
    """

    logger.info("Direct sun matrix for sensor grid")
    overwrite = config.getboolean("SimControl", "overwrite", fallback=False)
    pcdsmx = {}
    for grid_name, sender_grid in model.sender_grid.items():
        mpath.pcdsmx[grid_name] = Path("Matrices", f"pcdsmx_{grid_name}.mtx")
        if regen(mpath.pcdsmx[grid_name], config):
            logger.info("Generating using rcontrib...")
            rcvr_sun = matrix.sun_as_receiver(
                basis="r6",
                smx_path=mpath.smx,
                window_normals=model.window_normals,
                full_mod=True,
            )
            cdsmx_opt = config["SimControl"]["cdsmx_opt"]
            cdsmx_opt += f' -n {config["SimControl"]["nprocess"]}'
            cdsenv = [model.material_path, model.blackenvpath, *model.cfs_paths]
            sun_oct = Path("sun.oct")
            matrix.rcvr_oct(rcvr_sun, cdsenv, sun_oct)
            matrix.rcontrib(
                sender=sender_grid,
                modifier=rcvr_sun.modifier,
                octree=sun_oct,
                out=mpath.pcdsmx[grid_name],
                opt=cdsmx_opt,
            )
            sun_oct.unlink()


def direct_sun_matrix_vu(
    mpath: MradPath, model: MradModel, vmap_oct, cdmap_oct, config: ConfigParser
) -> None:
    """Compute direct sun matrix for images.
    Args:
        mpath:
        model:
        vmap_oct:
        cdmap_oct:
        config:
    Returns:
        None
    """
    logger.info("Direct sun matrix for view (image)")
    rcvr_sun = matrix.sun_as_receiver(
        basis="r6", smx_path=mpath.smx_sun_img, window_normals=model.window_normals
    )
    mod_names = [
        "%04d" % (int(line[3:]) - 1) for line in rcvr_sun.modifier.splitlines()
    ]
    sun_oct = Path("sun.oct")
    cdsenv = [model.material_path, model.blackenvpath, *model.cfs_paths]
    matrix.rcvr_oct(rcvr_sun, cdsenv, sun_oct)
    cdsmx_opt = config["SimControl"]["cdsmx_opt"]
    cdsmx_opt += f' -n {config["SimControl"]["nprocess"]}'
    for view, sndr in model.sender_view.items():
        mpath.vmap[view] = Path("Matrices", f"vmap_{view}.hdr")
        mpath.cdmap[view] = Path("Matrices", f"cdmap_{view}.hdr")
        vdict = model.views[view]
        vdict.pop("c", None)
        vdict.pop("pj", None)
        view_str = utils.opt2str(vdict)
        cmd = ["rpict"] + view_str.split()
        cmd += ["-ps", "1", "-ab", "0", "-av", ".31831", ".31831", ".31831"]
        cmd.append(vmap_oct)
        with open(mpath.vmap[view], "wb") as wtr:
            sp.run(cmd, check=True, stdout=wtr)
        cmd[-1] = cdmap_oct
        with open(mpath.cdmap[view], "wb") as wtr:
            sp.run(cmd, check=True, stdout=wtr)
        mpath.vcdfmx[view] = Path("Matrices", f"vcdfmx_{view}")
        mpath.vcdrmx[view] = Path("Matrices", f"vcdrmx_{view}")
        tempf = Path("Matrices", "vcdfmx")
        tempr = Path("Matrices", "vcdrmx")
        if regen(mpath.vcdfmx[view], config):
            logger.info(f"Generating direct sun f matrix for {view}")
            matrix.rcontrib(
                sender=sndr,
                modifier=rcvr_sun.modifier,
                octree=sun_oct,
                out=tempf,
                opt=cdsmx_opt,
            )
            mpath.vcdfmx[view].mkdir(exist_ok=True)
            for idx, file in enumerate(sorted(tempf.glob("*.hdr"))):
                file.rename(mpath.vcdfmx[view] / (mod_names[idx] + ".hdr"))
            shutil.rmtree(tempf)
        if regen(mpath.vcdrmx[view], config):
            logger.info(f"Generating direct sun r matrix for {view}")
            matrix.rcontrib(
                sender=sndr,
                modifier=rcvr_sun.modifier,
                octree=sun_oct,
                out=tempr,
                opt=cdsmx_opt,
            )
            mpath.vcdrmx[view].mkdir(exist_ok=True)
            for idx, file in enumerate(sorted(tempr.glob("*.hdr"))):
                file.rename(mpath.vcdrmx[view] / (mod_names[idx] + ".hdr"))
            shutil.rmtree(tempr)
    sun_oct.unlink()


def calc_2phase_pt(
    mpath: MradPath, model: MradModel, datetime_stamps: Sequence[str], config: ConfigParser
) -> None:
    """."""
    logger.info("Computing for 2-phase sensor grid results.")
    for grid_name in mpath.pdsmx:
        grid_lines = model.sender_grid[grid_name].sender.decode().strip().splitlines()
        # we don't care about the direction part
        xypos = [",".join(line.split()[:3]) for line in grid_lines]
        opath = Path("Results", f'grid_{config["Model"]["name"]}_{grid_name}.txt')
        res = mtxmult.mtxmult(mpath.pdsmx[grid_name], mpath.smx)
        if isinstance(res, bytes):
            res = res.decode().splitlines()
        else:
            res = ["\t".join(map(str, row)) for row in res.T.tolist()]
        with open(opath, "w") as wtr:
            wtr.write("\t" + "\t".join(xypos) + "\n")
            for idx, _ in enumerate(res):
                wtr.write(datetime_stamps[idx] + "\t")
                wtr.write(res[idx].rstrip() + "\n")


def calc_2phase_vu(mpath: MradPath, datetime_stamps, config: ConfigParser) -> None:
    """."""
    logger.info("Computing for 2-phase image-based results")
    for view in mpath.vdsmx:
        opath = Path("Results", f'view_{config["Model"]["name"]}_{view}')
        if opath.is_dir():
            shutil.rmtree(opath)
        utils.sprun(
            mtxmult.get_imgmult_cmd(
                mpath.vdsmx[view] / "%04d.hdr", mpath.smx, odir=opath
            )
        )
        ofiles = sorted(opath.glob("*.hdr"))
        for idx, val in enumerate(ofiles):
            val.rename(opath / (datetime_stamps[idx] + ".hdr"))


def calc_3phase_pt(
    mpath: MradPath,
    model: MradModel,
    datetime_stamps: list,
    config: ConfigParser,
):
    """."""
    logger.info("Computing for 3-phase sensor grid results")
    for grid_name in model.sender_grid:
        presl = []
        for wname in model.window_groups:
            _res = mtxmult.mtxmult(
                mpath.pvmx[grid_name + wname], model.bsdf_xml[wname], mpath.dmx[wname], mpath.smx
            )
            if isinstance(_res, bytes):
                presl.append(
                    [
                        map(float, line.decode().strip().split("\t"))
                        for line in _res.splitlines()
                    ]
                )
            else:
                presl.append(_res.T.tolist())
        res = [[sum(tup) for tup in zip(*line)] for line in zip(*presl)]
        respath = Path("Results", f'grid_{config["Model"]["name"]}_{grid_name}.txt')
        with open(respath, "w") as wtr:
            for idx, val in enumerate(res):
                wtr.write(datetime_stamps[idx] + ",")
                wtr.write(",".join(map(str, val)) + "\n")


def calc_3phase_vu(mpath: MradPath, model: MradModel, datetime_stamps, config: ConfigParser):
    """."""
    logger.info("Computing for 3-phase image-based results:")
    for view in model.sender_view:
        opath = Path("Results", f'view_{config["Model"]["name"]}_{view}')
        if opath.is_dir():
            shutil.rmtree(opath)
        logger.info("for %s", view)
        vresl = []
        for wname in model.window_groups:
            _vrespath = Path("Results", f"{view}_{wname}")
            _vrespath.mkdir(exist_ok=True)
            cmd = mtxmult.get_imgmult_cmd(
                mpath.vvmx[view + wname],
                model.bsdf_xml[wname],
                mpath.dmx[wname],
                mpath.smx,
                odir=_vrespath,
            )
            utils.sprun(cmd)
            vresl.append(_vrespath)
        if len(vresl) > 1:
            ops = ["+"] * (len(vresl) - 1)
            mtxmult.batch_pcomb(
                vresl, ops, opath, nproc=config.getint("SimControl", "nprocess")
            )
            for path in vresl:
                shutil.rmtree(path)
        else:
            shutil.move(vresl[0], opath)
        ofiles = sorted(opath.glob("*.hdr"))
        for idx, ofile in enumerate(ofiles):
            ofile.rename(opath / (datetime_stamps[idx] + ".hdr"))


def calc_5phase_pt(
    mpath: MradPath,
    model: MradModel,
    datetime_stamps: Sequence[str],
    config: ConfigParser,
) -> None:
    """."""
    logger.info("Computing sensor grid results")
    for grid_name in model.sender_grid:
        presl = []
        pdresl = []
        mult_cds = mtxmult.mtxmult(mpath.pcdsmx[grid_name], mpath.smx_sun)
        if isinstance(mult_cds, bytes):
            prescd = [
                list(map(float, line.decode().strip().split("\t")))
                for line in mult_cds.splitlines()
            ]
        else:
            prescd = mult_cds.T.tolist()
        for wname in model.window_groups:
            _res = mtxmult.mtxmult(
                mpath.pvmx[grid_name + wname], model.bsdf_xml[wname], mpath.dmx[wname], mpath.smx
            )
            _resd = mtxmult.mtxmult(
                mpath.pvmxd[grid_name + wname], model.bsdf_xml[wname], mpath.dmxd[wname], mpath.smxd
            )
            if isinstance(_res, bytes):
                _res = [
                    map(float, line.decode().strip().split("\t"))
                    for line in _res.splitlines()
                ]
                _resd = [
                    map(float, line.decode().strip().split("\t"))
                    for line in _resd.splitlines()
                ]
            else:
                _res = _res.T.tolist()
                _resd = _resd.T.tolist()
            presl.append(_res)
            pdresl.append(_resd)
        pres3 = [[sum(tup) for tup in zip(*line)] for line in zip(*presl)]
        pres3d = [[sum(tup) for tup in zip(*line)] for line in zip(*pdresl)]
        res = [
            [x - y + z for x, y, z in zip(a, b, c)]
            for a, b, c in zip(pres3, pres3d, prescd)
        ]
        respath = Path("Results", f'grid_{config["Model"]["name"]}_{grid_name}.txt')
        with open(respath, "w") as wtr:
            for idx in range(len(res)):
                wtr.write(datetime_stamps[idx] + ",")
                wtr.write(",".join(map(str, res[idx])) + "\n")


def calc_5phase_vu(
    mpath: MradPath,
    model: MradModel,
    datetime_stamps,
    datetime_stamps_d6,
    config: ConfigParser,
):
    """Compute for image-based 5-phase method result."""
    nprocess = config.getint("SimControl", "nprocess")
    for view in model.sender_view:
        logger.info(f"Computing for image-based results for {view}")
        vresl = []
        vdresl = []
        with tf.TemporaryDirectory() as td:
            vrescdr = Path(tf.mkdtemp(dir=td))
            vrescdf = Path(tf.mkdtemp(dir=td))
            vrescd = Path(tf.mkdtemp(dir=td))
            cmds = []
            cmds.append(
                mtxmult.get_imgmult_cmd(
                    mpath.vcdrmx[view] / "%04d.hdr", mpath.smx_sun_img, odir=vrescdr
                )
            )
            cmds.append(
                mtxmult.get_imgmult_cmd(
                    mpath.vcdfmx[view] / "%04d.hdr", mpath.smx_sun_img, odir=vrescdf
                )
            )
            for wname in model.window_groups:
                _vrespath = tf.mkdtemp(dir=td)
                _vdrespath = tf.mkdtemp(dir=td)
                cmds.append(
                    mtxmult.get_imgmult_cmd(
                        mpath.vvmx[view + wname],
                        model.bsdf_xml[wname],
                        mpath.dmx[wname],
                        mpath.smx,
                        odir=Path(_vrespath),
                    )
                )
                cmds.append(
                    mtxmult.get_imgmult_cmd(
                        mpath.vvmxd[view + wname],
                        model.bsdf_xml[wname],
                        mpath.dmxd[wname],
                        mpath.smxd,
                        odir=Path(_vdrespath),
                    )
                )
                vresl.append(Path(_vrespath))
                vdresl.append(Path(_vdrespath))
            logger.info("Multiplying matrices for images.")
            # i = 0
            for cmd in cmds:
                # i += 1
                # proc = sp.Popen(cmd)
                sp.run(cmd, check=True)
                # if i % nprocess == 0:
                    # proc.wait()
            # proc.wait()
            logger.info("Combine results for each window groups.")
            res3 = Path(tf.mkdtemp(dir=td))
            res3di = Path(tf.mkdtemp(dir=td))
            res3d = Path(tf.mkdtemp(dir=td))
            if len(model.window_groups) > 1:
                ops = ["+"] * (len(vresl) - 1)
                mtxmult.batch_pcomb(vresl, ops, res3, nproc=nprocess)
                mtxmult.batch_pcomb(vdresl, ops, res3di, nproc=nprocess)
            else:
                vresl[0].replace(res3)
                vdresl[0].replace(res3d)
            logger.info("Applying material reflectance map")
            mtxmult.batch_pcomb(
                [res3di, mpath.vmap[view]], ["*"], Path(res3d), nproc=nprocess
            )
            mtxmult.batch_pcomb(
                [vrescdr, mpath.cdmap[view], vrescdf],
                ["*", "+"],
                vrescd,
                nproc=nprocess,
            )
            opath = Path("Results", f"view_{config['Model']['name']}_{view}")
            if opath.is_dir():
                shutil.rmtree(opath)
            logger.info("Assemble all phase results.")
            res3_path = sorted(res3.glob("*.hdr"))
            [
                res3_path[idx].rename(res3 / (stamp + ".hdr"))
                for idx, stamp in enumerate(datetime_stamps)
            ]
            res3d_path = sorted(res3d.glob("*.hdr"))
            [
                res3d_path[idx].rename(res3d / (stamp + ".hdr"))
                for idx, stamp in enumerate(datetime_stamps)
            ]
            vrescd_path = sorted(vrescd.glob("*.hdr"))
            [
                vrescd_path[idx].rename(vrescd / (stamp + ".hdr"))
                for idx, stamp in enumerate(datetime_stamps_d6)
            ]
            opath.mkdir(exist_ok=True)
            ni = 0
            for hdr3 in os.listdir(res3):
                if hdr3 in os.listdir(vrescd):
                    ni += 1
                    cmd = [
                        "pcomb",
                        "-o",
                        str(res3 / hdr3),
                        "-s",
                        "-1",
                        "-o",
                        str(res3d / hdr3),
                        "-o",
                        str(vrescd / hdr3)]
                    with open(opath / hdr3, "wb") as wtr:
                        proc = sp.Popen(cmd, stdout=wtr)
                    if ni % nprocess == 0:
                        proc.wait()
                else:
                    os.rename(res3/hdr3, opath/hdr3)
            logger.info(f"Done computing for {view}")


def regen(path: Path, config) -> bool:
    """
    Decides whether to regenerate a file depending on
    if the file already exists and if overwrite is on.
    """
    if path.is_file():
        exist = True
    elif path.is_dir():
        exist = True
    else:
        exist = False
    return (not exist) or config.getboolean(
        "SimControl", "overwrite", fallback=False
    )


def two_phase(model: MradModel, config: ConfigParser) -> MradPath:
    """Two-phase simulation workflow."""
    mpath = MradPath()
    wea_meta, wea_data, wea_name = get_wea_data(config)
    mpath.smx = Path("Matrices") / (wea_name + ".smx")
    wea_data, datetime_stamps = sky.filter_wea(
        wea_data,
        wea_meta,
        daylight_hours_only=config.getboolean("Site", "daylight_hours_only"),
        start_hour=config.getfloat("Site", "start_hour"),
        end_hour=config.getfloat("Site", "end_hour"),
    )
    if regen(mpath.smx, config):
        sky.gendaymtx(
            mpath.smx, int(config["SimControl"]["smx_basis"][-1]), data=wea_data, meta=wea_meta
        )
    prep_2phase_pt(mpath, model, config)
    prep_2phase_vu(mpath, model, config)
    if not config.getboolean("SimControl", "no_multiply", fallback=False):
        calc_2phase_pt(mpath, model, datetime_stamps, config)
        calc_2phase_vu(mpath, datetime_stamps, config)
    os.remove(model.material_path)
    os.remove(model.blackenvpath)
    return mpath


def three_phase(model: MradModel, config: ConfigParser, direct=False) -> MradPath:
    """3/5-phase simulation workflow."""
    mpath = MradPath()
    wea_meta, wea_data, wea_name = get_wea_data(config)
    mpath.smx = Path("Matrices") / (wea_name + ".smx")
    wea_data, datetime_stamps = sky.filter_wea(
        wea_data,
        wea_meta,
        daylight_hours_only=config.getboolean("Site", "daylight_hours_only"),
        start_hour=config.getfloat("Site", "start_hour"),
        end_hour=config.getfloat("Site", "end_hour"),
    )
    if regen(mpath.smx, config):
        sky.gendaymtx(
            mpath.smx, int(config["SimControl"]["smx_basis"][-1]), data=wea_data, meta=wea_meta
        )
    view_matrix_pt(mpath, model, config)
    view_matrix_vu(mpath, model, config)
    daylight_matrix(mpath, model, config)
    if direct:
        wea_data_d6, datetime_stamps_d6 = sky.filter_wea(
            wea_data,
            wea_meta,
            daylight_hours_only=False,
            start_hour=0,
            end_hour=0,
            remove_zero=True,
            window_normals=model.window_normals,
        )
        mpath.smxd = Path("Matrices") / (wea_name + "_d.smx")
        sky.gendaymtx(
            mpath.smxd, int(config["SimControl"]["smx_basis"][-1]), data=wea_data, meta=wea_meta, direct=True
        )
        mpath.smx_sun_img = Path("Matrices") / (wea_name + "_d6_img.smx")
        sky.gendaymtx(
            mpath.smx_sun_img,
            int(config["SimControl"]["cdsmx_basis"][-1]),
            data=wea_data_d6,
            meta=wea_meta,
            onesun=True,
            direct=True,
        )
        mpath.smx_sun = Path("Matrices") / (wea_name + "_d6.smx")
        sky.gendaymtx(
            mpath.smx_sun,
            int(config["SimControl"]["cdsmx_basis"][-1]),
            data=wea_data,
            meta=wea_meta,
            onesun=True,
            direct=True,
        )
        vmap_oct, cdmap_oct = blacken_env(model, config)
        direct_sun_matrix_pt(mpath, model, config)
        direct_sun_matrix_vu(mpath, model, vmap_oct, cdmap_oct, config)
        daylight_matrix(mpath, model, config, direct=True)
        view_matrix_pt(mpath, model, config, direct=True)
        view_matrix_vu(mpath, model, config, direct=True)
        os.remove(vmap_oct)
        os.remove(cdmap_oct)
        os.remove(model.blackenvpath)
        if not config.getboolean("SimControl", "no_multiply", fallback=False):
            calc_5phase_pt(
                mpath,
                model,
                datetime_stamps,
                config,
            )
            calc_5phase_vu(
                mpath,
                model,
                datetime_stamps,
                datetime_stamps_d6,
                config,
            )
    else:
        if not config.getboolean("SimControl", "no_multiply", fallback=False):
            calc_3phase_pt(mpath, model, datetime_stamps, config)
            calc_3phase_vu(mpath, model, datetime_stamps, config)
    os.remove(model.material_path)
    return mpath


# def four_phase(model: MradModel, config: ConfigParser, direct=False):
#     """Four-phase simulation workflow."""
#     mpath = MradPath()
#     wea_meta, wea_data, wea_name = get_wea_data(config)
#     mpath.smx = Path("Matrices") / (wea_name + ".smx")
#     wea_meta, wea_data, datetime_stamps = sky.filter_wea(
#         wea_data,
#         wea_meta,
#         daylight_hours_only=config.getboolean("Site", "daylight_hours_only"),
#         start_hour=config.getfloat("Site", "start_hour"),
#         end_hour=config.getfloat("Site", "end_hour"),
#     )
#     if regen(mpath.smx, config):
#         sky.gendaymtx(
#             mpath.smx, config["SimControl"]["smx_basis"], data=wea_data, meta=wea_meta
#         )
#     view_matrix_pt(mpath, model, config)
#     view_matrix_vu(mpath, model, config)
#     fmxs = facade_matrix(model, config)
#     dmxs = daylight_matrix(model.port_prims, model, config)
#     if direct:
#         wea_path_d6, datetime_stamps_d6 = get_wea(
#             config, window_normals=model.window_normals
#         )
#         smxd = sky.gendaymtx(
#             wea_path, config["SimControl"]["smx_basis"], Path("Matrices"), direct=True
#         )
#         smx_sun_img = sky.gendaymtx(
#             wea_path_d6,
#             config["SimControl"]["cdsmx_basis"],
#             Path("Matrices"),
#             onesun=True,
#             direct=True,
#         )
#         smx_sun = sky.gendaymtx(
#             wea_path,
#             config["SimControl"]["cdsmx_basis"],
#             Path("Matrices"),
#             onesun=True,
#             direct=True,
#         )
#         vmap_oct, cdmap_oct = blacken_env(model, config)
#         pcdsmx = direct_sun_matrix_pt(mpath, model, smx_sun, config)
#         vcdfmx, vcdrmx, vmap_paths, cdmap_paths = direct_sun_matrix_vu(
#             mpath, model, smx_sun_img, vmap_oct, cdmap_oct, config
#         )
#         dmxsd = daylight_matrix(model.window_groups, model, config, direct=True)
#         pvmxsd = view_matrix_pt(mpath, model, config, direct=True)
#         vvmxsd = view_matrix_vu(mpath, model, config, direct=True)
#         calc_5phase_pt(
#             mpath,
#             model,
#             datetime_stamps,
#             config,
#         )
#         calc_5phase_vu(
#             mpath,
#             model,
#             datetime_stamps,
#             datetime_stamps_d6,
#             config,
#         )
#     else:
#         calc_4phase_pt(mpath, model, datetime_stamps, config)
        # calc_4phase_vu(vvmxs, fmxs, dmxs, smx)


#     def prep_4phase_vu(self):
#         """."""
#         vvmxs = {}
#         dmxs = {}
#         fmxs = {}
#         prcvr_windows = matrix.Receiver(
#             receiver='', basis=self.vmx_basis, modifier=None)
#         if len(self.sndr_views) > 0:
#             vrcvr_windows = {}
#             for view in model.sender_view:
#                 vrcvr_windows[view] = matrix.Receiver(
#                     receiver='', basis=self.vmx_basis, modifier=None)
#         port_prims = ncp.gen_ports_from_window_ncp
#             wpolys=window_prims, npolys=ncp_prims, depth=None, scale=None)
#         ncp.Genfmtx(win_polygons=window_polygon, port_prim=port_prims,
#                         out=kwargs['o'], env=kwargs['env'], sbasis=kwargs['ss'],
#                         rbasis=kwargs['rs'], opt=kwargs['opt'], refl=False,
#                         forw=False, wrap=False)
#         for wname in model.window_groups:
#             window_prim = model.window_groups[wname]
#             logger.info(f"Generating daylight matrix for {wname}")
#             dmxs[wname] = Path(self.mtxdir, f'dmx_{wname}.mtx')
#             sndr_window = matrix.Sender.as_surface(
#                 prim_list=window_prim, basis=self.vmx_basis, offset=None)
#             sndr_port = matrix.Sender.as_surface(
#                 prim_list=port_prims, basis=self.fmx_basis, offset=None)
#             matrix.rfluxmtx(sender=sndr_port, receiver=model.receiver_sky,
#                             env=self.envpath, out=dmxs[wname], opt=self.dmx_opt)
#             for view in model.sender_view:
#                 vvmxs[view+wname] = Path(
#                     self.mtxdir, f'vvmx_{view}_{wname}', '%04d.hdr')
#                 utils.mkdir_p(os.path.dirname(vvmxs[view+wname]))
#                 vrcvr_windows[view] += matrix.Receiver.as_surface(
#                     prim_list=window_prim, basis=self.vmx_basis,
#                     offset=None, left=None, source='glow', out=vvmxs[view+wname])
#         logger.info(f"Generating view matrix for {view}")
#         for view in model.sender_view:
#             matrix.rfluxmtx(sender=model.sender_view[view], receiver=vrcvr_windows[view],
#                             env=self.envpath, opt=self.vmx_opt, out=None)
