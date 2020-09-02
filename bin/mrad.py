#!/usr/bin/env python3
"""
T.Wang

"""

import argparse
from configparser import ConfigParser
import logging
import os
import shutil
from frads import mtxmethod
from frads import radutil

def initialize():
    templ = mtxmethod.cfg_template
    fs = templ['FileStructure']
    fs['base'] = os.getcwd()
    if fs['objects'] in os.listdir(fs['base']):
        files = os.listdir(os.path.join(fs['base'], fs['objects']))
        objfiles = [f for f in files if f.endswith('.rad')]
        matfiles = [f for f in files if f.endswith('.mat')]
        templ['Model']['scene'] = ' '.join(objfiles)
        templ['Model']['material'] = ' '.join(matfiles)
    else:
        radutil.mkdir_p(fs['objects'])
    radutil.mkdir_p(fs['matrices'])
    radutil.mkdir_p(fs['results'])
    radutil.mkdir_p(fs['resources'])
    cfg = ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    cfg.read_dict(templ)
    with open("run.cfg", 'w') as rdr:
        cfg.write(rdr)

def main(cfgpath):
    cfg = ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    with open(cfgpath) as rdr:
        cfg.read_string(rdr.read())
    setup = mtxmethod.Prepare(cfg)
    mrad = mtxmethod.MTXmethod(setup)
    ncp_shade = setup.model['ncp_shade']
    if setup.model['bsdf'] is None:
        logger.info("Using two-phase method")
        mrad.prep_2phase()
        mrad.calc_2phase()
    else:
        if setup.simctrl.getboolean('separate_direct'):
            if ncp_shade is not None and len(ncp_shade.split()) > 1:
                logger.info('Using six-phase simulation')
                mrad.prep_6phase()
                mrad.calc_6phase()
            else:
                logger.info('Using five-phase simulation')
                mrad.prep_5phase()
                mrad.calc_5phase()
        else:
            if ncp_shade is not None and len(ncp_shade.split()) > 1:
                logger.info('Using four-phase simulation')
                mrad.prep_4phase()
                mrad.calc_4phase()
            else:
                logger.info('Using three-phase simulation')
                mrad.prep_3phase()
                mrad.calc_3phase()


if __name__ == '__main__':
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
        main(args.cfg)
    else:
        raise Exception("init or run")
