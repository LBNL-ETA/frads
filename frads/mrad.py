"""
Executive command-line program for Radiance matrix-based simulation.
"""

from configparser import ConfigParser
import argparse
import glob
import logging
import os
from frads import mtxmethod
from frads import util

logger = logging.getLogger('frads')


def mkdirs(cfg: util.MradConfig) -> None:
    """Silently make directories based on configuration."""
    util.mkdir_p(cfg.objdir)
    util.mkdir_p(cfg.mtxdir)
    util.mkdir_p(cfg.resdir)
    util.mkdir_p(cfg.rsodir)


def initialize(args: argparse.Namespace) -> None:
    """Initiate mrad operation.
    Going through files in the standard file
    structure and generate a default.cfg file.
    Args:
        args: argparse.Namespace
    """
    cwd = os.getcwd()
    file_struct = {'base': args.base, 'objects': args.objdir,
                   'matrices': args.mtxdir, 'resources': args.rsodir,
                   'results': args.resdir}
    model = {'material': '', 'scene': '', 'window_paths': '',
             'window_xml': '', 'window_cfs': ''}
    raysender = {'grid_surface': args.grid[0], 'grid_spacing': args.grid[1],
                 'grid_height': args.grid[2], 'view': ''}
    if (args.latlon == ('','')) and (args.wea_path == '') and (args.zipcode == ''):
        raise ValueError("Site not defined, use --wea_path | --latlon | --zipcode")
    site = {'wea_path':args.wea_path, 'latitude': args.latlon[0],
            'longitude':args.latlon[1], 'zipcode':args.zipcode}
    object_pattern: str = args.object if args.object is not None else '.rad'
    window_pattern: str = args.window if args.window is not None else 'window*.rad'
    material_pattern: str = args.material if args.material is not None else '*.mat'
    if args.objdir in os.listdir(args.base):
        os.chdir(os.path.join(args.base, args.objdir))
        window_files = sorted(glob.glob(window_pattern))
        all_obj_files = glob.glob(object_pattern)
        obj_files = [f for f in all_obj_files if f not in window_files]
        material_files = glob.glob(material_pattern)
        model['scene'] = ' '.join(obj_files)
        model['material'] = ' '.join(material_files)
        model['window_paths'] = ' '.join(window_files)
    else:
        logger.warning("No %s directory found at %s, making so",
                       args.obj, args.base)
        util.mkdir_p(os.path.join(args.base, args.obj))
    util.mkdir_p(os.path.join(args.base, args.mtxdir))
    util.mkdir_p(os.path.join(args.base, args.resdir))
    util.mkdir_p(os.path.join(args.base, args.rsodir))
    cfg = ConfigParser(allow_no_value=True)
    templ_config = {"File Structure": file_struct, "Site": site,
                    "Model": model, "Ray Sender": raysender}
    cfg.read_dict(templ_config)
    os.chdir(cwd)
    with open("default.cfg", 'w') as rdr:
        cfg.write(rdr)


def convert_config(cfg: ConfigParser) -> util.MradConfig:
    """Convert a configparser object into a dictionary.
    Args:
        cfg: SafeConfigParser
    Returns:
        MradConfig object (dataclass)
    """
    cfg_dict = {}
    sections = cfg.sections()
    for sec in sections:
        cfg_dict.update(dict(cfg[sec]))
    for key, value in cfg_dict.items():
        if value is not None:
            if value.lower() == 'true':
                cfg_dict[key] = True
            elif value.lower() == 'false':
                cfg_dict[key] = False
        else:
            cfg_dict[key] = ''
    if cfg_dict['scene'] is None:
        raise ValueError("No scene description")
    return util.MradConfig(**cfg_dict)


def run(args: argparse.Namespace) -> None:
    """Call mtxmethod to carry out the actual simulation."""
    cfg = ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    with open(args.cfg) as rdr:
        cfg.read_string(rdr.read())
    config = convert_config(cfg)
    config.name = util.basename(args.cfg)
    mkdirs(config)
    model = mtxmethod.assemble_model(config)
    if config.method != '':
        _method = getattr(mtxmethod, config.method)
        _method(model, config, direct=config.separate_direct)
    else:
        if '' in (config.window_xml, config.windows):
            logger.info("Using two-phase method")
            mtxmethod.two_phase(model, config)
        else:
            if config.ncp_shade != '' and len(config.ncp_shade.split()) > 1:
                if config.separate_direct:
                    logger.info('Using six-phase simulation')
                    mtxmethod.four_phase(model, config, direct=True)
                else:
                    logger.info('Using four-phase simulation')
                    mtxmethod.four_phase(model, config)
            else:
                if config.separate_direct:
                    logger.info('Using five-phase simulation')
                    mtxmethod.three_phase(model, config, direct=True)
                else:
                    logger.info('Using three-phase simulation')
                    mtxmethod.three_phase(model, config)


def main() -> None:
    """Parse the configuration file and envoke to mtxmethod to do the actual work."""
    parser = argparse.ArgumentParser(
        prog='mrad', description='Executive program for Radiance matrix-based simulation')
    subparser = parser.add_subparsers()
    parser_init = subparser.add_parser('init')
    parser_init.set_defaults(func=initialize)
    parser_init.add_argument("-B", "--base", default=os.getcwd())
    parser_init.add_argument("-O", "--objdir", default='Objects')
    parser_init.add_argument("-M", "--mtxdir", default='Matrices')
    parser_init.add_argument("-S", "--rsodir", default='Resources')
    parser_init.add_argument("-R", "--resdir", default='Results')
    parser_init.add_argument("-W", "--wea_path", default='')
    parser_init.add_argument("-Z", "--zipcode", default='')
    parser_init.add_argument("-L", "--latlon", nargs=2, default=('',''))
    parser_init.add_argument("-o", "--object")
    parser_init.add_argument("-m", "--material")
    parser_init.add_argument("-w", "--window")
    parser_init.add_argument("-g", "--grid", nargs=3, default=('', 0, 0))
    parser_run = subparser.add_parser('run')
    parser_run.add_argument('cfg', help='configuration file path')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: 1=Debug; 2=Info; 3=Warning; 4=Error; 5=Critical. \
        E.g. -vvv=Warning, default=%(default)s')
    parser_run.set_defaults(func=run)
    args = parser.parse_args()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    _level = args.verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    args.func(args)
