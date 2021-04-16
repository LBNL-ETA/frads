"""
Executive program for Radiance matrix-based simulation.
"""

import argparse
from configparser import ConfigParser
import logging
import os
from frads import mtxmethod
from frads import radutil


def initialize():
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
    """Convert a configparser object to dictionary."""
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


def main():
    """Parse the configuration file and envoke to mtxmethod to do the actual work."""
    parser = argparse.ArgumentParser()
    parser.add_argument('op')
    parser.add_argument('cfg', nargs='?', default='run.cfg')
    parser.add_argument('-vb', action='store_true', help='verbose mode')
    parser.add_argument('-db', action='store_true', help='debug mode')
    parser.add_argument('-si', action='store_true', help='silent mode')
    args = parser.parse_args()
    logger = logging.getLogger('frads')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    _level = logging.WARNING
    if args.db:
        _level = logging.DEBUG
    elif args.vb:
        _level = logging.INFO
    elif args.si:
        _level = logging.CRITICAL
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if args.op == 'init':
        initialize()
    elif args.op == 'run':
        cfg = ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
        with open(args.cfg) as rdr:
            cfg.read_string(rdr.read())
        cfg_dict = cfg2dict(cfg)
        msetup = mtxmethod.MTXMethod(cfg_dict)
        ncp_shade = msetup.config.ncp_shade
        smx = msetup.gen_smx(msetup.config.smx_basis)
        if msetup.config.method is not None:
            _method = globals()[msetup.config.method]
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
    else:
        raise Exception("init or run")


