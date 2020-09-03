#!/usr/bin/env python3
"""
T.Wang

"""

import argparse
from configparser import ConfigParser
import logging
import os
from frads import mtxmethod
from frads import radutil

import pdb

def initialize():
    """."""
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

def cfg2dict(cfg):
    """."""
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


def main(cfgpath):
    """."""
    cfg = ConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    with open(cfgpath) as rdr:
        cfg.read_string(rdr.read())
    cfg_dict = cfg2dict(cfg)
    msetup = mtxmethod.MTXMethod(cfg_dict)
    ncp_shade = msetup.config.ncp_shade
    smx = msetup.gen_smx(msetup.config.smx_basis)
    if msetup.config.bsdf is None:
        logger.info("Using two-phase method")
        pdsmx = msetup.prep_2phase_pt()
        vdsmx = msetup.prep_2phase_vu()
        msetup.calc_2phase_pt(pdsmx, smx)
        msetup.calc_2phase_vu(vdsmx, smx)
    else:
        if msetup.config.separate_direct:
            if ncp_shade is not None and len(ncp_shade.split()) > 1:
                logger.info('Using six-phase simulation')
                msetup.prep_6phase_pt()
                msetup.prep_6phase_vu()
            else:
                smx_d = msetup.gen_smx(self.dmx_basis, direct=True)
                logger.info('Using five-phase simulation')
                msetup.prep_5phase_pt()
                msetup.prep_5phase_vu()
                msetup.calc_5phase_pt(vmx, vmxd, dmx, dmxd, csmx, smx, smxd, smx_sun)
                msetup.calc_5phase_vu()
        else:
            if ncp_shade is not None and len(ncp_shade.split()) > 1:
                logger.info('Using four-phase simulation')
                msetup.prep_4phase_pt()
                msetup.calc_4phase_vu()
            else:
                logger.info('Using three-phase simulation')
                dmxs = msetup.prep_3phase_dmx()
                pvmxs = msetup.prep_3phase_pt()
                vvmxs = msetup.prep_3phase_vu()


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
