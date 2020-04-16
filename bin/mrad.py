#!/usr/bin/env python3
"""
T.Wang

"""

import argparse
from configparser import ConfigParser
from frads import mtxmethod
import shutil
import pdb


def main(cfgpath):
    setup = mtxmethod.Prepare(cfgpath)
    mrad = mtxmethod.MTXmethod(setup)
    ncp_shade = setup.model['ncp_shade']
    if setup.model['bsdf'] is None:
        logger.info("Using two-phase method")
        mrad.prep_2phase()
        mrad.calc_2phase()
    else:
        if setup.simctrl['separate_direct'].lower() == 'true':
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
    shutil.rmtree(mrad.td)


if __name__ == '__main__':
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('cfgpath')
    parser.add_argument('-vb', action='store_true', help='verbose mode')
    parser.add_argument('-db', action='store_true', help='debug mode')
    parser.add_argument('-si', action='store_true', help='silent mode')
    args = parser.parse_args()
    argmap = vars(args)
    logger = logging.getLogger('frads')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if argmap['db']:
        logger.setLevel(logging.DEBUG)
    elif argmap['vb']:
        logger.setLevel(logging.INFO)
    elif argmap['si']:
        logger.setLevel(logging.CRITICAL)
    main(args.cfgpath)
