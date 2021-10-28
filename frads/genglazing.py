"""Generate glazing+shading system (BSDF) using WINDOW.

This is a self-contained program that uses pywincalc to compute
window system properties. pywincalc is a python wrapper for wincalc,
which is a c++ wrapper for wincalc_engine, the engine behind LBNL
WINDOW software and EnergyPlus window calculation module.
Thus, pywincalc is a dependency for this tool, but is not checked when
installing frads.
"""

import argparse
import configparser
from dataclasses import dataclass
import os
import pywincalc as pwc
import subprocess as sp
from typing import List, NamedTuple

from frads import util


@dataclass
class GlazingSystem:
    system: List[str]
    optic_standards: str = ''
    spacing: float = 0.2
    depth: float = 0.3
    tilt: float = 40
    curve: float = 0
    material: str = ''
    blind1: str = ''
    blind2: str = ''
    blind3: str = ''
    shade1: str = ''
    shade2: str = ''
    shade3: str = ''
    glazing1: str = ''
    glazing2: str = ''
    glazing3: str = ''
    glazing4: str = ''
    gap1: str = ''
    gap2: str = ''
    gap3: str = ''
    width: float = 1
    height: float = 1

    def __post_init__(self):
        if self.glazing1 == '':
            raise ValueError("No glazing defined")


def get_igsdb_json(igsdb_id, token, xml=False):
    """Get igsdb data by igsdb_id"""
    if token is None:
        raise ValueError("Need IGSDB token")
    url = "https://igsdb.lbl.gov/api/v1/products/{}"
    if xml:
        url += "/datafile"
    header = {"Authorization": "Token " + token}
    response = util.request(url.format(igsdb_id), header=header)
    if response == '{"detail":"Not found."}':
        raise ValueError("Unknown igsdb id: ", igsdb_id)
    return response


def compose_blinds(spacing, width, angle, curve, material):
    """."""
    if material == '':
        return None
    number_segments = 7  # Not sure if this does anything
    geometry = pwc.VenetianGeometry(
        width, spacing, curve, angle, number_segments)
    composition_data = pwc.ProductComposistionData(material, geometry)
    venetian_layer = pwc.ComposedProductData(composition_data)
    venetian_solid_layer = pwc.convert_to_solid_layer(venetian_layer)
    return venetian_solid_layer


def get_arg_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='genglazing', usage='genglazing inp',
        description='Generate a glazing system using pwc')

    parser.add_argument("inp")
    parser.add_argument("-M", '--mtx')
    parser.add_argument("-T", "--key")
    return parser


def convert_config(cfg: configparser.ConfigParser) -> GlazingSystem:
    """."""
    config_dict = {}
    config_dict.update(dict(cfg['Blinds']))
    config_dict.update(dict(cfg['Shade']))
    config_dict.update(dict(cfg['Glazing']))
    config_dict.update(dict(cfg['Gap']))
    config_dict['width'] = float(cfg['Glazing system']['width'])
    config_dict['height'] = float(cfg['Glazing system']['height'])
    config_dict['system'] = cfg['Glazing system']['system'].split()
    return GlazingSystem(**config_dict)


def initialize():
    """Write a configration file for users' convenience."""
    blinds = {'blind1': None, 'blind2': None, 'blind4': None, }
    shade = {'shade1': None, 'shade2': None, 'shade3': None, }
    glazing = {'glazing1': None, 'glazing2': None, 'glazing3': None, }
    gap = {'gap1': None, 'gap2': None, 'gap3': None, 'gap4': None, }
    glazing_system = {'width': 1, 'height': 1, 'system': None}
    templ_config = {
        "Blind": blinds,
        "Shade": shade,
        "Glazing": glazing,
        "Gap": gap,
        "System": glazing_system,
    }
    cfg = configparser.ConfigParser(allow_no_value=True)
    cfg.read_dict(templ_config)
    with open("default_glazing_system.cfg", 'w') as rdr:
        cfg.write(rdr)


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    config = configparser.ConfigParser(
        allow_no_value=True, inline_comment_prefixes='#')
    config.read(args.inp)
    glzsys = convert_config(config)
    system_layer = []
    gap_layer = []
    for layer in glzsys.system[1::2]:
        _layer = getattr(glzsys, layer).split()
        thickness: float = float(_layer[0])
        if len(_layer) > 2:
            gas_comp = _layer[1:int((len(_layer) - 1) / 2 + 1)]
            gas_perc = _layer[int((len(_layer) - 1) / 2 + 1):]
            _gap = []
            for gas, perc in zip(gas_comp, gas_perc):
                _gas = getattr(pwc.PredefinedGasType, gas.upper())
                _gap.append(pwc.PredefinedGasMixtureComponent(
                    _gas, float(perc)))
            gap_layer.append(pwc.Gap(_gap, thickness))
        else:
            gap_layer.append(pwc.Gap(
                getattr(pwc.PredefinedGasType, _layer[1].upper()),
                float(thickness)))
    BSDF_calc = False if not args.mtx else True
    for layer in glzsys.system[::2]:
        layer_id = getattr(glzsys, layer)
        if (not os.path.isfile(layer_id)) and (layer_id[0].isdigit()):
            xml = False
            if layer.startswith('shade'):
                xml = True
                BSDF_calc = True
                ret = get_igsdb_json(layer_id, args.key, xml=xml)
                system_layer.append(pwc.parse_bsdf_xml_string(ret))
            elif layer.startswith('blind'):
                spacing, depth, tilt, curve, material_id = layer_id.split()
                xml = True
                BSDF_calc = True
                material = pwc.parse_json(get_igsdb_json(material_id, args.key))
                system_layer.append(compose_blinds(
                    float(spacing), float(depth), float(tilt), float(curve), material))
            else:
                ret = get_igsdb_json(layer_id, args.key, xml=xml)
                system_layer.append(pwc.parse_json(ret))
        elif not layer_id.endswith('xml'):
            system_layer.append(pwc.parse_optics_file(layer_id))
        elif layer_id.endswith('xml'):
            system_layer.append(pwc.parse_bsdf_xml_file(layer_id))
        else:
            system_layer.append(pwc.parse_optics_file(layer_id))

    _file_path_ = os.path.dirname(__file__)
    standard_fname = "W5_NFRC_2003.std"
    standard_path = os.path.join(_file_path_, 'data', 'standards', standard_fname)
    if not os.path.isfile(standard_path):
        raise FileNotFoundError(standard_path)
    glazing_shgc_env = {
        'optical_standard': pwc.load_standard(standard_path),
        'solid_layers': system_layer,
        'gap_layers': gap_layer,
        'width_meters': glzsys.width,
        'height_meters': glzsys.height,
        'environment': pwc.nfrc_shgc_environments(),
    }
    if BSDF_calc:
        glazing_shgc_env['bsdf_hemisphere'] = pwc.BSDFHemisphere.create(
            pwc.BSDFBasisType.FULL)
    glzsys_shgc_env = pwc.GlazingSystem(**glazing_shgc_env)
    shgc = glzsys_shgc_env.shgc()
    solar_optical_results = glzsys_shgc_env.optical_method_results("SOLAR")
    visible_optical_results = glzsys_shgc_env.optical_method_results("PHOTOPIC")
    print("SHGC: ", shgc)
    print("Tsol: ", solar_optical_results.system_results.front.transmittance.direct_hemispherical)
    print("Tvis: ", visible_optical_results.system_results.front.transmittance.direct_hemispherical)
    for idx, layer in enumerate(solar_optical_results.layer_results):
        print(f"fAbs{idx}: ", layer.front.absorptance.direct)
        print(f"bAbs{idx}: ", layer.back.absorptance.direct)
    if args.mtx:
        t_f = visible_optical_results.system_results.front.transmittance.matrix
        t_b = visible_optical_results.system_results.back.transmittance.matrix
        r_f = visible_optical_results.system_results.front.reflectance.matrix
        r_b = visible_optical_results.system_results.back.reflectance.matrix
        if t_f is not None:
            with open('tmp_tf', 'w') as wtr:
                [wtr.write(' '.join(map(str, row))+'\n') for row in t_f]
            with open('tmp_tb', 'w') as wtr:
                [wtr.write(' '.join(map(str, row))+'\n') for row in t_b]
            with open('tmp_rf', 'w') as wtr:
                [wtr.write(' '.join(map(str, row))+'\n') for row in r_f]
            with open('tmp_rb', 'w') as wtr:
                [wtr.write(' '.join(map(str, row))+'\n') for row in r_b]
            cmd = ['wrapBSDF', '-W', '-a', 'kf', '-tf', 'tmp_tf', '-tb',
                   'tmp_tb', '-rf', 'tmp_rf', '-rb', 'tmp_rb', '-U']
            wrap_bsdf = sp.run(cmd, stdout=sp.PIPE)
            with open(args.mtx, 'wb') as wtr:
                wtr.write(wrap_bsdf.stdout)
