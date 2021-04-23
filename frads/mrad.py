"""
Executive command-line program for Radiance matrix-based simulation.
"""

from configparser import ConfigParser
import argparse
import logging
import os
from frads import mtxmethod
from frads import radutil


def mkdirs(cfg):
    """Make directories according to the configuration dict."""
    base = cfg['base']
    objdir = os.path.join(base, cfg['objects'])
    mtxdir = os.path.join(base, cfg['matrices'])
    resdir = os.path.join(base, cfg['results'])
    resodir = os.path.join(base, cfg['resources'])
    radutil.mkdir_p(objdir)
    radutil.mkdir_p(mtxdir)
    radutil.mkdir_p(resdir)
    radutil.mkdir_p(resodir)

def initialize(args):
    """Going through files in the standard file structure and generate a cfg file."""
    templ = mtxmethod.cfg_template
    templ['base'] = os.getcwd()
    if templ['objects'] in os.listdir(templ['base']):
        files = os.listdir(os.path.join(templ['base'], templ['objects']))
        objfiles = [f for f in files if f.endswith('.rad')]
        matfiles = [f for f in files if f.endswith('.mat')]
        templ['scene'] = ' '.join(objfiles)
        templ['material'] = ' '.join(matfiles)
    else:
        radutil.mkdir_p(templ['objects'])
    radutil.mkdir_p(templ['matrices'])
    radutil.mkdir_p(templ['results'])
    radutil.mkdir_p(templ['resources'])
    cfg = ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    templ = {"mrad configration":templ}
    cfg.read_dict(templ)
    with open("run.cfg", 'w') as rdr:
        cfg.write(rdr)


def cfg2dict(cfg):
    """Convert a configparser object into a dictionary."""
    cfg_dict = {}
    sections = cfg.sections()
    for sec in sections:
        cfg_dict.update(dict(cfg[sec]))
    for k,v in cfg_dict.items():
        if v is not None:
            if v.lower() == 'true':
                cfg_dict[k] = True
            elif v.lower() == 'false':
                cfg_dict[k] = False
            elif v == '':
                cfg_dict[k] = None
    return cfg_dict

def run(args):
    """Call mtxmethod to carry out the actual simulation."""
    cfg = ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    with open(args.cfg) as rdr:
        cfg.read_string(rdr.read())
    cfg_dict = cfg2dict(cfg)
    mkdirs(cfg_dict)
    msetup = mtxmethod.MTXMethod(cfg_dict)
    ncp_shade = msetup.config.ncp_shade
    smx = msetup.gen_smx(msetup.wea_path, msetup.config.smx_basis, msetup.mtxdir)
    if msetup.config.method is not None:
        _method = getattr(mtxmethod, msetup.config.method)
        _method(msetup, smx, direct=msetup.config.separate_direct)
    else:
        if None in (msetup.config.bsdf, msetup.config.windows):
            logger.info("Using two-phase method")
            mtxmethod.two_phase(msetup, smx)
        else:
            if ncp_shade is not None and len(ncp_shade.split()) > 1:
                if msetup.config.separate_direct:
                    logger.info('Using six-phase simulation')
                    mtxmethod.four_phase(msetup, smx, direct=True)
                else:
                    logger.info('Using four-phase simulation')
                    mtxmethod.four_phase(msetup, smx)
            else:
                if msetup.config.separate_direct:
                    logger.info('Using five-phase simulation')
                    mtxmethod.three_phase(msetup, smx, direct=True)
                else:
                    logger.info('Using three-phase simulation')
                    mtxmethod.three_phase(msetup, smx)

def main():
    """Parse the configuration file and envoke to mtxmethod to do the actual work."""
    global logger
    parser = argparse.ArgumentParser(
        prog='mrad', description='Executive program for carry out Radiance matrix-based simulation')
    subparser = parser.add_subparsers()
    parser_init = subparser.add_parser('init')
    parser_init.set_defaults(func=initialize)
    parser_run = subparser.add_parser('run')
    parser_run.add_argument('cfg', default='run.cfg', help='configuration file path')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: 1=Debug; 2=Info; 3=Warning; 4=Error; 5=Critical. E.g. -vvv=Warning, default=%(default)s')
    parser_run.set_defaults(func=run)
    args = parser.parse_args()
    logger = logging.getLogger('frads')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    _level = args.verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    args.func(args)
