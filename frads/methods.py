"""Typical Radiance matrix-based simulation workflows
"""

from configparser import ConfigParser
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import shutil
import subprocess as sp
import tempfile as tf
from typing import List
from typing import Dict
from typing import Generator
from typing import Sequence
from typing import Tuple

from frads import geom
from frads import sky
from frads import matrix

from frads import mtxmult
from frads import parsers
from frads import raycall
from frads import utils
from frads.types import Options
from frads.types import Primitive
from frads.types import Receiver
from frads.types import Sender
from frads.types import MradModel
from frads.types import MradPath
from frads.types import WeaMetaData
from frads.types import WeaData


logger: logging.Logger = logging.getLogger("frads.methods")


def get_window_group(wpaths: List[Path], orientation=None) -> Tuple[dict, list]:
    """Parse window groups from config.

    Args:
        wpaths(str): window file paths
    Return:
        window_groups(dict): window group name and primitives
        window_normals(list): a list of normal for each window group.
    """
    window_groups = {}
    window_normals: List[geom.Vector] = []
    for wpath in wpaths:
        prims = utils.unpack_primitives(wpath)
        window_groups[wpath.stem] = prims
        # Taking normal from the first polygon
        _normal = parsers.parse_polygon(prims[0].real_arg).normal
        if _normal not in window_normals:
            window_normals.append(_normal)
    final_window_normals = window_normals
    if (orientation is not None) and (float(orientation) != 0):
        rorient = float(orientation) * 180 / 3.14159265358579
        final_window_normals = [n.rotate_3d(geom.Vector(0, 0, 1), rorient)
                                for n in window_normals]
    return window_groups, final_window_normals


def get_ncp_shades(npaths: List[Path]) -> dict:
    """Parse ncp shade groups from config."""
    ncp_shades = {}
    for npath in npaths:
        prims = utils.unpack_primitives(npath)
        ncp_shades[npath.stem] = prims
    return ncp_shades


def get_wea_data(config: ConfigParser) -> Tuple[WeaMetaData, List[WeaData], str]:
    """Get wea data and parse into appropriate data types."""
    if wea_path := config["Site"].getpath("wea_path"):
        logger.info("Using user specified %s file.", wea_path)
        name = wea_path.stem
        with open(wea_path, "r", encoding="utf-8") as rdr:
            wea_metadata, wea_data = parsers.parse_wea(rdr.read())
    elif epw_path := config["Site"].getpath("epw_path"):
        logger.info("Converting %s to a .wea file", epw_path)
        name = epw_path.stem
        with open(epw_path, "r", encoding="utf-8") as rdr:
            wea_metadata, wea_data = parsers.parse_epw(rdr.read())
    else:
        raise ValueError("Need either a .wea or a .epw file")
    return wea_metadata, wea_data, name


def get_sender_grid(config: ConfigParser) -> Dict[str, Sender]:
    """Get point grid as ray senders."""
    sender_grid: Dict[str, Sender] = {}
    if (grid_paths := config["RaySender"].getpaths("grid_surface")) is None:
        return sender_grid
    for gpath in grid_paths:
        name: str = gpath.stem
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
            config["RaySender"].getfloat("grid_height"),
            config["RaySender"].getfloat("grid_spacing"),
        )
        sender_grid[name] = matrix.points_as_sender(
            pts_list=sensor_pts, ray_cnt=config["SimControl"].getint("ray_count")
        )
    return sender_grid


def get_sender_view(config: ConfigParser) -> Tuple[dict, dict]:
    """Get a single view as a sender.
    Args:
        config: MradConfig object"""
    sender_view: Dict[str, matrix.Sender] = {}
    view_dicts: Dict[str, View] = {}
    if (view := config["RaySender"].getview("view")) is None:
        return sender_view, view_dicts
    view_name = "view_00"
    # if "vf" in vdict:
    # with open(vdict["vf"], "r", encoding="ascii") as rdr:
    # vdict.update(parsers.parse_vu(rdr.read()))
    view_dicts[view_name] = view
    sender_view[view_name] = matrix.view_as_sender(
        view=view,
        ray_cnt=int(config["SimControl"]["ray_count"]),
        xres=view.xres,
        yres=view.yres,
    )
    return sender_view, view_dicts


@contextmanager
def assemble_model(config: ConfigParser) -> Generator:
    """Assemble all the pieces together."""
    logger.info("Model assembling")
    # Get Ray senders
    sender_grid = get_sender_grid(config)
    sender_view, view_dicts = get_sender_view(config)
    if (not sender_grid) and (not sender_view):
        raise ValueError("Need to at least specify a grid or a view")
    # Get materials
    material_primitives = []
    for path in config["Model"].getpaths("material"):
        for prim in utils.unpack_primitives(path):
            material_primitives.append(prim)
    material_primitives.append(
        Primitive("void", "plastic", "black", ["0"], [5, 0, 0, 0, 0, 0])
    )
    material_primitives.append(
        Primitive("void", "glow", "glowing", ["0"], [4, 1, 1, 1, 0])
    )
    material_path = Path(f"all_material_{utils.id_generator()}.rad")
    with open(material_path, "w", encoding="ascii") as wtr:
        for primitive in material_primitives:
            wtr.write(str(primitive) + "\n")
    # Orientation
    orientation = config["Site"].get("orientation")
    # Get window groups
    window_groups, window_normals = get_window_group(
        config["Model"].getpaths("windows", []), orientation=orientation
    )
    # Get BSDFs
    bsdf_mat = {
        wname: Path(path)
        for wname, path in zip(
            window_groups, config["Model"].getpaths("window_xmls", [])
        )
    }
    # Get ncp shades
    ncp_shades = get_ncp_shades(config["Model"].getpaths("ncps", []))
    # Get cfs paths
    cfs_path = config["Model"].getpaths("window_cfs", [])
    black_env_path = Path(f"blackened_{utils.id_generator()}.rad")
    with open(black_env_path, "w", encoding="ascii") as wtr:
        for path in config["Model"].getpaths("scene"):
            wtr.write(f"\n!xform -m black {path}")
    yield MradModel(
        config["Model"].get("name"),
        material_path,
        window_groups,
        window_normals,
        sender_grid,
        sender_view,
        view_dicts,
        bsdf_mat,
        cfs_path,
        ncp_shades,
        black_env_path,
    )
    logger.info("Cleaning up")
    os.remove(material_path)
    os.remove(black_env_path)


def prep_2phase_pt(mpath: MradPath, model: MradModel, config: ConfigParser) -> None:
    """Prepare matrices two phase methods."""
    logger.info("Computing for 2-phase sensor point matrices...")
    sys_paths = [
        model.material_path,
        *config["Model"].getpaths("scene"),
        *config["Model"].getpaths("windows"),
    ]
    opt = config["SimControl"].getoptions("dsmx_opt")
    opt["n"] = config["SimControl"].getint("nprocess")
    overwrite = config.getboolean("SimControl", "overwrite", fallback=False)
    for grid_name, sender_grid in model.sender_grid.items():
        mpath.pdsmx[grid_name] = Path("Matrices", f"pdsmx_{model.name}_{grid_name}.mtx")
        receiver_sky = matrix.sky_as_receiver(
            config["SimControl"]["smx_basis"],
            out=mpath.pdsmx[grid_name],
        )
        if (not mpath.pdsmx[grid_name].is_file()) or overwrite:
            matrix.rfluxmtx(sender_grid, receiver_sky, sys_paths, utils.opt2list(opt))


def prep_2phase_vu(mpath: MradPath, model: MradModel, config: ConfigParser) -> None:
    """Generate image-based matrices if view defined."""
    logger.info("Computing for image-based 2-phase matrices...")
    sys_paths = [
        model.material_path,
        *config["Model"].getpaths("scene"),
        *config["Model"].getpaths("windows"),
    ]
    opt = config["SimControl"].getoptions("dsmx_opt")
    opt["n"] = config["SimControl"].getint("nprocess")
    overwrite = config.getboolean("SimControl", "overwrite", fallback=False)
    for view_name, sender_view in model.sender_view.items():
        mpath.vdsmx[view_name] = Path(
            "Matrices", f"vdsmx_{model.name}_{view_name}", "%04d.hdr"
        )
        receiver_sky = matrix.sky_as_receiver(
            config["SimControl"]["smx_basis"],
            out=mpath.vdsmx[view_name],
        )
        if (not mpath.vdsmx[view_name].is_dir()) or overwrite:
            logger.info("Generating for %s", view_name)
            matrix.rfluxmtx(sender_view, receiver_sky, sys_paths, utils.opt2list(opt))


def view_matrix_pt(
    mpath: MradPath, model: MradModel, config: ConfigParser, direct: bool = False
) -> None:
    """."""
    _opt = config["SimControl"].getoptions("vmx_opt")
    _env: List[Path] = [
        model.material_path,
        *config["Model"].getpaths("scene"),
    ]
    if direct:
        logger.info("Computing direct view matrix for sensor grid:")
        _opt["ab"] = 1
        _env = [model.material_path, model.black_env_path]
    else:
        logger.info("Computing view matrix for sensor grid:")
    receiver_windows = Receiver(receiver="", basis=config["SimControl"]["vmx_basis"])
    for grid_name, sender_grid in model.sender_grid.items():
        for wname, window_prim in model.window_groups.items():
            _name = grid_name + wname
            if direct:
                mpath.pvmxd[_name] = Path(
                    "Matrices", f"pvmx_{model.name}_{_name}_d.mtx"
                )
                out = mpath.pvmxd[_name]
            else:
                mpath.pvmx[_name] = Path("Matrices", f"pvmx_{model.name}_{_name}.mtx")
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
            files_exist = all(f.is_file() for f in mpath.pvmxd.values())
        else:
            files_exist = all(f.is_file() for f in mpath.pvmx.values())
        if (not files_exist) or config.getboolean(
            "SimControl", "overwrite", fallback=False
        ):
            logger.info("Generating vmx for %s", grid_name)
            matrix.rfluxmtx(sender_grid, receiver_windows, _env, utils.opt2list(_opt))


def view_matrix_vu(
    mpath: MradPath, model: MradModel, config: ConfigParser, direct: bool = False
) -> None:
    """Prepare matrices using three-phase methods."""
    _opt = config["SimControl"].getoptions("vmx_opt")
    _env = [model.material_path, *config["Model"].getpaths("scene")]
    direct_msg = ""
    if direct:
        _opt["i"] = True
        _opt["ab"] = 1
        _env = [model.material_path, model.black_env_path]
        direct_msg = " direct-only"
    _opt["n"] = config["SimControl"].getint("nprocess")
    overwrite = config.getboolean("SimControl", "overwrite", fallback=False)
    for view, sender_view in model.sender_view.items():
        vrcvr_windows = Receiver(receiver="", basis=config["SimControl"]["vmx_basis"])
        for wname, window_prim in model.window_groups.items():
            _name = view + wname
            if direct:
                mpath.vvmxd[_name] = Path(
                    "Matrices", f"vvmx_{model.name}_{_name}_d", "%04d.hdr"
                )
                out = mpath.vvmxd[_name]
            else:
                mpath.vvmx[_name] = Path(
                    "Matrices", f"vvmx_{model.name}_{_name}", "%04d.hdr"
                )
                out = mpath.vvmx[_name]
            out.parent.mkdir(exist_ok=True)
            vrcvr_windows += matrix.surface_as_receiver(
                prim_list=window_prim,
                basis=config["SimControl"]["vmx_basis"],
                source="glow",
                out=out,
            )
        if direct:
            exists = all(
                any(d.iterdir()) for d in mpath.vvmxd[_name].parents[1].glob("vvmx*_d")
            )
        else:
            exists = all(
                any(d.iterdir())
                for d in mpath.vvmx[_name].parents[1].glob("vvmx*[!_d]")
            )
        if (not exists) or overwrite:
            logger.info("Computing%s image-based view matrix", direct_msg)
            matrix.rfluxmtx(sender_view, vrcvr_windows, _env, utils.opt2list(_opt))


def daylight_matrix(
    mpath: MradPath, model: MradModel, config: ConfigParser, direct: bool = False
) -> None:
    """Call rfluxmtx to generate daylight matrices for each sender surface."""
    logger.info("Computing daylight matrix...")
    dmx_opt = config["SimControl"].getoptions("dmx_opt")
    dmx_opt["n"] = config["SimControl"].getint("nprocess")
    dmx_env = [model.material_path, *config["Model"].getpaths("scene")]
    if direct:
        dmx_opt["ab"] = 0
        dmx_env = [model.material_path, model.black_env_path]
    for sname, surface_primitives in model.window_groups.items():
        _name = sname
        if direct:
            mpath.dmxd[sname] = Path("Matrices", f"dmx_{model.name}_{_name}_d.mtx")
            out = mpath.dmxd[sname]
        else:
            mpath.dmx[sname] = Path("Matrices", f"dmx_{model.name}_{_name}.mtx")
            out = mpath.dmx[sname]
        if regen(out, config):
            logger.info("Generating daylight matrix for %s", _name)
            sndr_window = matrix.surface_as_sender(
                prim_list=surface_primitives, basis=config["SimControl"]["vmx_basis"]
            )
            receiver_sky = matrix.sky_as_receiver(
                config["SimControl"]["smx_basis"], out
            )
            matrix.rfluxmtx(sndr_window, receiver_sky, dmx_env, utils.opt2list(dmx_opt))


def blacken_env(model: MradModel, config: ConfigParser) -> Tuple[str, str]:
    """."""
    bwindow_path = f"blackened_window_{utils.id_generator()}.rad"
    gwindow_path = f"glowing_window_{utils.id_generator()}.rad"
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
    with open(bwindow_path, "w", encoding="ascii") as wtr:
        wtr.write("\n".join(list(map(str, blackened_window))))
    with open(gwindow_path, "w", encoding="ascii") as wtr:
        wtr.write("\n".join(list(map(str, glowing_window))))
    vmap_oct = f"vmap_{utils.id_generator()}.oct"
    cdmap_oct = f"cdmap_{utils.id_generator()}.oct"
    raycall.oconv(
        str(model.material_path),
        *map(str, config["Model"].getpaths("scene")),
        gwindow_path,
        outpath=vmap_oct,
        frozen=True,
    )
    logger.info("Generating view matrix material map octree")
    raycall.oconv(
        str(model.material_path),
        *map(str, config["Model"]["scene"].split()),
        bwindow_path,
        outpath=cdmap_oct,
        frozen=True,
    )
    logger.info("Generating direct-sun matrix material map octree")
    os.remove(bwindow_path)
    os.remove(gwindow_path)
    return vmap_oct, cdmap_oct


def direct_sun_matrix_pt(
    mpath: MradPath, model: MradModel, config: ConfigParser
) -> None:
    """Compute direct sun matrix for sensor points.
    Args:
        smx_path: path to sun only sky matrix
    Returns:
        path to resulting direct sun matrix
    """

    logger.info("Direct sun matrix for sensor grid")
    for grid_name, sender_grid in model.sender_grid.items():
        mpath.pcdsmx[grid_name] = Path(
            "Matrices", f"pcdsmx_{model.name}_{grid_name}.mtx"
        )
        if regen(mpath.pcdsmx[grid_name], config):
            logger.info("Generating using rcontrib...")
            rcvr_sun = matrix.sun_as_receiver(
                basis="r6",
                smx_path=mpath.smx_sun,
                window_normals=model.window_normals,
                full_mod=True,
            )
            cdsmx_opt = config["SimControl"].getoptions("cdsmx_opt")
            cdsmx_opt["n"] = config["SimControl"].getint("nprocess")
            cdsenv = [model.material_path, model.black_env_path, *model.cfs_paths]
            sun_oct = Path(f"sun_{utils.id_generator()}.oct")
            matrix.rcvr_oct(rcvr_sun, cdsenv, sun_oct)
            matrix.rcontrib(
                sender_grid,
                rcvr_sun.modifier,
                sun_oct,
                mpath.pcdsmx[grid_name],
                utils.opt2list(cdsmx_opt),
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
    mod_names = [f"{int(line[3:]):04d}" for line in rcvr_sun.modifier.splitlines()]
    sun_oct = Path(f"sun_{utils.id_generator()}.oct")
    cdsenv = [model.material_path, model.black_env_path, *model.cfs_paths]
    matrix.rcvr_oct(rcvr_sun, cdsenv, sun_oct)
    cdsmx_opt = config["SimControl"].getoptions("cdsmx_opt")
    cdsmx_opt["n"] = config["SimControl"].getint("nprocess")
    cdsmx_opt_list = utils.opt2list(cdsmx_opt)
    for view, sndr in model.sender_view.items():
        mpath.vmap[view] = Path("Matrices", f"vmap_{model.name}_{view}.hdr")
        mpath.cdmap[view] = Path("Matrices", f"cdmap_{model.name}_{view}.hdr")
        # view = model.views[view]
        # vdict.pop("c", None)
        # vdict.pop("pj", None)
        rpict_opt = Options()
        rpict_opt.ps = 1
        rpict_opt.ab = 0
        rpict_opt.av = (0.31831, 0.31831, 0.31831)
        cmd = raycall.get_rpict_command(model.views[view], rpict_opt, octree=vmap_oct)
        logger.info("Generating view matrix material map with: \n %s", " ".join(cmd))
        utils.run_write(cmd, mpath.vmap[view])
        cmd[-1] = cdmap_oct
        logger.info(
            "Generating direct-sun matrix material map with: \n %s", " ".join(cmd)
        )
        utils.run_write(cmd, mpath.cdmap[view])
        mpath.vcdfmx[view] = Path("Matrices", f"vcdfmx_{model.name}_{view}")
        mpath.vcdrmx[view] = Path("Matrices", f"vcdrmx_{model.name}_{view}")
        tempf = Path("Matrices", "vcdfmx")
        tempr = Path("Matrices", "vcdrmx")
        if regen(mpath.vcdfmx[view], config):
            logger.info("Generating direct sun f matrix for %s", view)
            matrix.rcontrib(sndr, rcvr_sun.modifier, sun_oct, tempf, cdsmx_opt_list)
            mpath.vcdfmx[view].mkdir(exist_ok=True)
            for idx, file in enumerate(sorted(tempf.glob("*.hdr"))):
                file.replace(mpath.vcdfmx[view] / (mod_names[idx] + ".hdr"))
            shutil.rmtree(tempf)
        if regen(mpath.vcdrmx[view], config):
            logger.info("Generating direct sun r matrix for %s", view)
            cdsmx_opt_list.append("-i+")
            matrix.rcontrib(sndr, rcvr_sun.modifier, sun_oct, tempr, cdsmx_opt_list)
            mpath.vcdrmx[view].mkdir(exist_ok=True)
            for idx, file in enumerate(sorted(tempr.glob("*.hdr"))):
                file.replace(mpath.vcdrmx[view] / (mod_names[idx] + ".hdr"))
            shutil.rmtree(tempr)
    sun_oct.unlink()


def calc_2phase_pt(
    mpath: MradPath,
    model: MradModel,
    datetime_stamps: Sequence[str],
) -> None:
    """."""
    logger.info("Computing for 2-phase sensor grid results.")
    for grid_name in mpath.pdsmx:
        grid_lines = model.sender_grid[grid_name].sender.decode().strip().splitlines()
        # we don't care about the direction part
        xypos = [",".join(line.split()[:3]) for line in grid_lines]
        opath = Path("Results", f"grid_{model.name}_{grid_name}.txt")
        res = mtxmult.mtxmult(mpath.pdsmx[grid_name], mpath.smx)
        if isinstance(res, bytes):
            res = res.decode().splitlines()
        else:
            res = ["\t".join(map(str, row)) for row in res.T.tolist()]
        with open(opath, "w", encoding="utf-8") as wtr:
            wtr.write("\t" + "\t".join(xypos) + "\n")
            for idx, value in enumerate(res):
                wtr.write(datetime_stamps[idx] + "\t")
                wtr.write(value.rstrip() + "\n")


def calc_2phase_vu(mpath: MradPath, model: MradModel, datetime_stamps) -> None:
    """."""
    logger.info("Computing for 2-phase image-based results")
    for view in mpath.vdsmx:
        opath = Path("Results", f"view_{model.name}_{view}")
        if opath.is_dir():
            shutil.rmtree(opath)
        cmd = mtxmult.get_imgmult_cmd(mpath.vdsmx[view], mpath.smx, odir=opath)
        logger.info(" ".join(cmd))
        sp.run(cmd, check=True)
        ofiles = sorted(opath.glob("*.hdr"))
        for idx, val in enumerate(ofiles):
            val.replace(opath / (datetime_stamps[idx] + ".hdr"))


def calc_3phase_pt(
    mpath: MradPath,
    model: MradModel,
    datetime_stamps: list,
) -> None:
    """."""
    logger.info("Computing for 3-phase sensor grid results")
    for grid_name in model.sender_grid:
        presl = []
        grid_lines = model.sender_grid[grid_name].sender.decode().strip().splitlines()
        xyzpos = [",".join(line.split()[:3]) for line in grid_lines]
        for wname in model.window_groups:
            _res = mtxmult.mtxmult(
                mpath.pvmx[grid_name + wname],
                model.bsdf_xml[wname],
                mpath.dmx[wname],
                mpath.smx,
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
        respath = Path("Results", f"grid_{model.name}_{grid_name}.txt")
        with open(respath, "w", encoding="utf-8") as wtr:
            wtr.write("\t" + "\t".join(xyzpos) + "\n")
            for idx, val in enumerate(res):
                wtr.write(datetime_stamps[idx] + "\t")
                wtr.write("\t".join(map(str, val)) + "\n")


def calc_3phase_vu(
    mpath: MradPath, model: MradModel, datetime_stamps, config: ConfigParser
) -> None:
    """."""
    for view in model.sender_view:
        opath = Path("Results", f"view_{model.name}_{view}")
        if opath.is_dir():
            shutil.rmtree(opath)
        logger.info("Computing for 3-phase image-based results for %s", view)
        vresl = []
        for wname in model.window_groups:
            _vrespath = Path("Results", f"{view}_{model.name}_{wname}")
            _vrespath.mkdir(exist_ok=True)
            cmd = mtxmult.get_imgmult_cmd(
                mpath.vvmx[view + wname],
                model.bsdf_xml[wname],
                mpath.dmx[wname],
                mpath.smx,
                odir=_vrespath,
            )
            logger.info(" ".join(cmd))
            sp.run(cmd, check=True)
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
            ofile.replace(opath / (datetime_stamps[idx] + ".hdr"))


def calc_5phase_pt(
    mpath: MradPath,
    model: MradModel,
    datetime_stamps: Sequence[str],
) -> None:
    """."""
    logger.info("Computing sensor grid results")
    for grid_name in model.sender_grid:
        presl = []
        pdresl = []
        grid_lines = model.sender_grid[grid_name].sender.decode().strip().splitlines()
        xyzpos = [",".join(line.split()[:3]) for line in grid_lines]
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
                mpath.pvmx[grid_name + wname],
                model.bsdf_xml[wname],
                mpath.dmx[wname],
                mpath.smx,
            )
            _resd = mtxmult.mtxmult(
                mpath.pvmxd[grid_name + wname],
                model.bsdf_xml[wname],
                mpath.dmxd[wname],
                mpath.smxd,
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
        respath = Path("Results", f"grid_{model.name}_{grid_name}.txt")
        with open(respath, "w", encoding="utf-8") as wtr:
            wtr.write("\t" + "\t".join(xyzpos) + "\n")
            for idx, val in enumerate(res):
                wtr.write(datetime_stamps[idx] + "\t")
                wtr.write("\t".join(map(str, val)) + "\n")


def calc_5phase_vu(
    mpath: MradPath,
    model: MradModel,
    datetime_stamps,
    datetime_stamps_d6,
    config: ConfigParser,
) -> None:
    """Compute for image-based 5-phase method result."""
    nprocess = config.getint("SimControl", "nprocess")
    for view in model.sender_view:
        logger.info("Computing for image-based results for %s", view)
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
            for cmd in cmds:
                logger.info(" ".join(cmd))
            utils.batch_process(cmds, nproc=nprocess)
            logger.info("Combine results for each window groups.")
            res3 = Path(tf.mkdtemp(dir=td))
            res3di = Path(tf.mkdtemp(dir=td))
            res3d = Path(tf.mkdtemp(dir=td))
            if len(model.window_groups) > 1:
                ops = ["+"] * (len(vresl) - 1)
                mtxmult.batch_pcomb(vresl, ops, res3, nproc=nprocess)
                mtxmult.batch_pcomb(vdresl, ops, res3di, nproc=nprocess)
            else:
                for file in vresl[0].glob("*.hdr"):
                    file.replace(res3 / file.name)
                for file in vdresl[0].glob("*.hdr"):
                    file.replace(res3di / file.name)
            logger.info("Applying material reflectance map")
            mtxmult.batch_pcomb(
                [res3di, mpath.vmap[view]], ["*"], res3d, nproc=nprocess
            )
            mtxmult.batch_pcomb(
                [vrescdr, mpath.cdmap[view], vrescdf],
                ["*", "+"],
                vrescd,
                nproc=nprocess,
            )
            opath = Path("Results", f"view_{model.name}_{view}")
            if opath.is_dir():
                shutil.rmtree(opath)
            logger.info("Assemble all phase results.")
            res3_path = sorted(res3.glob("*.hdr"))
            for idx, stamp in enumerate(datetime_stamps):
                res3_path[idx].replace(res3 / (stamp + ".hdr"))
            res3d_path = sorted(res3d.glob("*.hdr"))
            for idx, stamp in enumerate(datetime_stamps):
                res3d_path[idx].replace(res3d / (stamp + ".hdr"))
            vrescd_path = sorted(vrescd.glob("*.hdr"))
            for idx, stamp in enumerate(datetime_stamps_d6):
                vrescd_path[idx].replace(vrescd / (stamp + ".hdr"))
            opath.mkdir(exist_ok=True)
            cmds = []
            opaths = []
            for hdr3 in os.listdir(res3):
                if hdr3 in os.listdir(vrescd):
                    opaths.append(opath / hdr3)
                    cmds.append(
                        [
                            "pcomb",
                            "-o",
                            str(res3 / hdr3),
                            "-s",
                            "-1",
                            "-o",
                            str(res3d / hdr3),
                            "-o",
                            str(vrescd / hdr3),
                        ]
                    )
                else:
                    os.replace(res3 / hdr3, opath / hdr3)
            if len(cmds) > 0:
                utils.batch_process(cmds, opaths=opaths)
            logger.info("Done computing for %s", view)


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
    return (not exist) or config.getboolean("SimControl", "overwrite", fallback=False)


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
            mpath.smx,
            int(config["SimControl"]["smx_basis"][-1]),
            data=wea_data,
            meta=wea_meta,
            rotate=config["Site"]["orientation"],
        )
    prep_2phase_pt(mpath, model, config)
    prep_2phase_vu(mpath, model, config)
    if not config.getboolean("SimControl", "no_multiply", fallback=False):
        calc_2phase_pt(mpath, model, datetime_stamps)
        calc_2phase_vu(mpath, model, datetime_stamps)
    return mpath


def three_phase(
    model: MradModel, config: ConfigParser, direct: bool = False
) -> MradPath:
    """3/5-phase simulation workflow."""
    do_multiply = config.getboolean("SimControl", "no_multiply", fallback=False)
    psteps = 12 if direct else 4
    if do_multiply:
        psteps += 2
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
            mpath.smx,
            int(config["SimControl"]["smx_basis"][-1]),
            data=wea_data,
            meta=wea_meta,
            rotate=config["Site"]["orientation"],
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
            mpath.smxd,
            int(config["SimControl"]["smx_basis"][-1]),
            data=wea_data,
            meta=wea_meta,
            direct=True,
        )
        mpath.smx_sun_img = Path("Matrices") / (wea_name + "_d6_img.smx")
        sky.gendaymtx(
            mpath.smx_sun_img,
            int(config["SimControl"]["cdsmx_basis"][-1]),
            data=wea_data_d6,
            meta=wea_meta,
            rotate=config["Site"]["orientation"],
            onesun=True,
            direct=True,
        )
        mpath.smx_sun = Path("Matrices") / (wea_name + "_d6.smx")
        sky.gendaymtx(
            mpath.smx_sun,
            int(config["SimControl"]["cdsmx_basis"][-1]),
            data=wea_data,
            meta=wea_meta,
            rotate=config["Site"]["orientation"],
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
        if not do_multiply:
            calc_5phase_pt(
                mpath,
                model,
                datetime_stamps,
            )
            calc_5phase_vu(
                mpath,
                model,
                datetime_stamps,
                datetime_stamps_d6,
                config,
            )
    else:
        if not do_multiply:
            calc_3phase_pt(mpath, model, datetime_stamps)
            calc_3phase_vu(mpath, model, datetime_stamps, config)
    return mpath
