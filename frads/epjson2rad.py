"""
Convert an EnergyPlus model as a parsed dictionary
to Radiance primitives.
Todo:
    * Parse window data file for Construction:WindowDataFile

"""
import argparse
from collections import namedtuple
from configparser import ConfigParser
import json
import logging
import os
from typing import List
import subprocess as sp
from frads import radutil as ru
from frads import radgeom as rg
from frads import util

logger = logging.getLogger("frads.epjson2rad")

PI = 3.14159265358579

# Sensor grid dimension in meters
GRID_HEIGHT = 0.75
GRID_SPACING = 0.6

Primitive = ru.Primitive


def read_epjs(fpath: str) -> dict:
    """Load and parse input file into a JSON object.
    If the input file is in .idf fomart, use command-line
    energyplus program to convert it to epJSON format
    Args:
        fpath: input file path
    Returns:
        epjs: JSON object as a Python dictionary
    """
    epjson_path: str = ''
    if fpath.endswith('.idf'):
        cmd = ['energyplus', '--convert-only', fpath]
        sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        epjson_path = os.path.splitext(fpath)[0]+'.epJSON'
        if not os.path.isfile(epjson_path):
            raise OSError("idf to epjson conversion failed")
    elif fpath.endswith('.epJSON'):
        epjson_path = fpath
    with open(epjson_path) as rdr:
        epjs = json.load(rdr)
    return epjs


class epJSON2Rad:
    """
    Convert EnergyPlus JSON objects into Radiance primitives.
    """
    material = namedtuple("material", ["primitive", "thickness"])
    matrices = {}

    def __init__(self, epjs):
        """Parse EPJSON object passed in as a dictionary."""
        self.epjs = epjs
        self.checkout_fene(epjs)
        self.get_material_prims(epjs)
        self.zones = self.get_zones(epjs)
        self.site = self.get_site(epjs)

    def thicken(self, surface, windows, thickness):
        """Thicken window-wall."""
        direction = surface.normal().scale(thickness)
        facade = surface.extrude(direction)[:2]
        [facade.extend(window.extrude(direction)[2:]) for window in windows]
        uniq = facade.copy()
        for idx, val in enumerate(facade):
            for rep in facade[:idx]+facade[idx+1:]:
                if set(val.to_list()) == set(rep.to_list()):
                    uniq.remove(rep)
        return uniq

    def _material(self, epjs_mat):
        """Parse EP Material"""
        mat_prims = {}
        for key, val in epjs_mat.items():
            try:
                refl = 1 - val['visible_absorptance']
            except KeyError as ke:
                logger.warning(ke)
                logger.warning(f"No visible absorptance defined for {key}, assuming 50%")
                refl = .5
            _real_args = "5 {0:.2f} {0:.2f} {0:.2f} 0 0".format(refl)
            try:
                _thickness = val['thickness']
            except KeyError as ke:
                logger.info(f"{key} has zero thickness")
                _thickness = 0
            mat_prims[key] = self.material(
                Primitive('void', 'plastic', key.replace(' ','_'), '0', _real_args),
                _thickness)
        return mat_prims

    def _material_nomass(self, epjs_mat):
        """Parse EP Material:NoMass"""
        mat_prims = {}
        for key, val in epjs_mat.items():
            _identifier = _modifier = _type = _real_arg = ''
            if 'visible_absorptance' in val:
                _modifier = 'void'
                _type = 'plastic'
                refl = 1 - val['visible_absorptance']
                _identifier = key.replace(' ','_')
                _real_arg = "5 {0:.2f} {0:.2f} {0:.2f} 0 0".format(refl)
                _thickness = 0
            mat_prims[key] = self.material(
                Primitive(_modifier, _type, _identifier, '0', _real_arg),
                0) # zero thickness
        return mat_prims

    def _windowmaterial_gap(self, epjs_wndw_mat):
        window_material_primitives = {}
        for material in epjs_wndw_mat:
            window_material_primitives[material]= self.material(
                Primitive('void', 'glass', material, '0', '3 1 1 1'),
                epjs_wndw_mat[material]['thickness'])
        return window_material_primitives

    def _windowmaterial_gas(self, epjs_wndw_mat):
        window_material_primitives = {}
        for material in epjs_wndw_mat:
            window_material_primitives[material]= self.material(
                Primitive('void', 'glass', material, '0', '3 1 1 1'),
                epjs_wndw_mat[material]['thickness'])
        return window_material_primitives


    def _windowmaterial_simpleglazingsystem(self, epjs_wndw_mat):
        """Parse EP WindowMaterial:Simpleglazing"""
        wndw_mat_prims = {}
        for key, val in epjs_wndw_mat.items():
            try:
                tmis = util.tmit2tmis(val['visible_transmittance'])
            except KeyError as ke:
                raise Exception(ke, "for", key)
            wndw_mat_prims[key] = self.material(
                Primitive('void', 'glass', key.replace(' ','_'), '0',
                "3 {0:.2f} {0:.2f} {0:.2f}".format(tmis)), 0.06)
        return wndw_mat_prims

    def _windowmaterial_simpleglazing(self, epjs_wndw_mat):
        """Parse EP WindowMaterial:Simpleglazing"""
        wndw_mat_prims = {}
        for key, val in epjs_wndw_mat.items():
            try:
                tmis = util.tmit2tmis(val['visible_transmittance'])
            except KeyError as ke:
                print(key)
                raise ke
            wndw_mat_prims[key] = self.material(
                Primitive('void', 'glass', key.replace(' ','_'), '0',
                "3 {0:.2f} {0:.2f} {0:.2f}".format(tmis)), 0.06)
        return wndw_mat_prims

    def _windowmaterial_glazing(self, epjs_wndw_mat):
        """Parse EP WindowMaterial:Glazing"""
        wndw_mat_prims = {}
        for key, val in epjs_wndw_mat.items():
            if val['optical_data_type'].lower() == 'bsdf':
                pass
            else:
                tvis = val['visible_transmittance_at_normal_incidence']
                tmis = util.tmit2tmis(tvis)
                wndw_mat_prims[key] = self.material(
                    Primitive('void', 'glass', key.replace(' ','_'), '0',
                    "3 {0:.2f} {0:.2f} {0:.2f}".format(tmis)), 0.06)
        return wndw_mat_prims

    def _windowmaterial_blind(self, blind_dict: dict) -> dict:
        """Parse EP WindowMaterial:Blind"""
        blind_prims = {}
        for key, val in blind_dict.items():
            _id = key.replace(' ','_')
            back_beam_vis_refl = val['back_side_slat_beam_visible_reflectance']
            back_diff_vis_refl = val['back_side_slat_diffuse_visible_reflectance']
            front_beam_vis_refl = val['front_side_slat_beam_visible_reflectance']
            front_diff_vis_refl = val['front_side_slat_diffuse_visible_reflectance']
            slat_width = val['slat_width']
            slat_thickness = val['slat_thickness']
            slat_separation = val['slat_separation']
            slat_angle = val['slat_angle']
            blind_prims[key] = Primitive(
                'void', 'plastic', _id, '0',
                '5 {0:.2f} {0:.2f} {0:.2f} 0 0'.format(front_diff_vis_refl))
            genblinds_cmd = f"genblinds {_id} {_id} {slat_width} 3 {20*slat_separation} {slat_angle}"
        return blind_prims


    def get_material_prims(self, epjs):
        """Call the corresponding parser for each material type."""
        mkeys = [key for key in epjs.keys() if 'material' in key.split(':')[0].lower()]
        self.mat_prims = {}
        for key in mkeys:
            tocall = getattr(self, f"_{key.replace(':', '_')}".lower())
            self.mat_prims.update(tocall(epjs[key]))


    def checkout_fene(self, epjs):
        """
        Check if the model has any window.
        Record the host of the each window.
        """
        try:
            self.fene_srfs = epjs['FenestrationSurface:Detailed']
        except KeyError as ke:
            raise Exception(ke, 'not found, no exterior window found.')
        self.fene_hosts = set([val['building_surface_name']
                               for key, val in self.fene_srfs.items()])

    def check_ext_window(self, zone_srfs):
        """Check if the window is connected to exterior."""
        has_window = False
        for key, val in zone_srfs.items():
            if (key in self.fene_hosts) and (val['sun_exposure']!='NoSun'):
                has_window = True
                break
        return has_window

    def parse_cnstrct(self, cnstrct):
        """Parse the construction data."""
        layers: List[str] = ['outside_layer']
        layers.extend(sorted([l for l in cnstrct if l.startswith('layer_')]))
        inner_layer = cnstrct[layers[-1]]
        outer_layer = cnstrct[layers[0]].replace(' ','_')
        cthick = sum([self.mat_prims[cnstrct[l]].thickness for l in layers])
        return inner_layer, outer_layer, cthick


    def parse_wndw_cnstrct(self, wcnstrct):
        """Parse window construction."""
        if wcnstrct['ctype'] == 'default':
            material_name = wcnstrct['outside_layer'].replace(" ", "_")
        elif wcnstrct['ctype'] == 'cfs':
            material_name = wcnstrct['visible_optical_complex_back_transmittance_matrix_name']
        else:
            raise ValueError("Unknown construction: ", wcnstrct)
        return material_name

    def check_srf_normal(self, zone):
        """Check the surface normal in each zone."""
        polygons = [ru.parse_polygon(v.real_arg) for v in zone['Floor'].values()]
        polygons.extend([ru.parse_polygon(v.real_arg) for v in zone['Ceiling'].values()])
        polygons.extend([ru.parse_polygon(v.real_arg) for v in zone['Wall'].values()])
        polygons.extend([ru.parse_polygon(v.real_arg) for v in zone['Window'].values()])
        centroid = rg.polygon_center(*polygons)
        new_floor = {}
        for key, val in zone['Floor'].items():
            fpolygon = ru.parse_polygon(val.real_arg)
            angle2center = fpolygon.normal().angle_from(centroid - fpolygon.centroid())
            if angle2center < PI/4:
                new_prim = Primitive(val.modifier, val.ptype, val.identifier,
                                     '0', fpolygon.flip().to_real())
                new_floor[key] = new_prim
            else:
                new_floor[key] = val
        zone['Floor'] = new_floor
        new_window = {}
        for key, val in zone['Window'].items():
            wpolygon = ru.parse_polygon(val.real_arg)
            angle2center = wpolygon.normal().angle_from(centroid - wpolygon.centroid())
            if angle2center > PI/4:
                new_prim = Primitive(val.modifier, val.ptype, val.identifier,
                                     '0', wpolygon.flip().to_real())
                new_window[key] = new_prim
            else:
                new_window[key] = val
        zone['Window'] = new_window
        return zone

    def _construction_windowdatafile(self, construction):
        file_path = construction['file_name']
        # self.parse_wndw_data(file_path)
        # return layers, cthick

    def _construction_complexfenestrationstate(self, cnstrct):
        for key, val in cnstrct.items():
            val['ctype'] = "cfs"
            tf_name = val['visible_optical_complex_front_transmittance_matrix_name']
            tb_name = val['visible_optical_complex_back_transmittance_matrix_name']
            tf_list = self.epjs['Matrix:TwoDimension'][tf_name]['values']
            tb_list = self.epjs['Matrix:TwoDimension'][tb_name]['values']
            ncolumn = self.epjs['Matrix:TwoDimension'][tf_name]['number_of_columns']
            tf_bsdf = util.nest_list([v['value'] for v in tf_list], ncolumn)
            tb_bsdf = util.nest_list([v['value'] for v in tb_list], ncolumn)
            tf = ru.BSDFData(tf_bsdf).to_sdata()
            tb = ru.BSDFData(tb_bsdf).to_sdata()
            self.matrices[key] = ru.RadMatrix(tf, tb)
        return cnstrct

    def _construction(self, cnstrct):
        for val in cnstrct.values():
            val['ctype'] = 'default'
        return cnstrct

    def get_construction(self, epjs) -> dict:
        keys = [key for key in epjs.keys()
                if 'construction' in key.split(':')[0].lower()]
        construction = {}
        for key in keys:
            tocall = getattr(self, f"_{key.replace(':', '_')}".lower())
            construction.update(tocall(epjs[key]))
        return construction

    def get_zones(self, epjs):
        """Looping through zones."""
        opaque_srfs = epjs['BuildingSurface:Detailed']
        construction = self.get_construction(epjs)
        ext_zones = {}
        for zone_name in epjs['Zone']:
            zone_srfs = {k:v for k,v in opaque_srfs.items()
                         if v['zone_name'] == zone_name}
            if self.check_ext_window(zone_srfs):
                zone = {'Wall':{}, 'Floor':{}, 'Ceiling':{},
                        'Window':{}, 'Roof':{}}
                for sn in zone_srfs:
                    surface_name = sn.replace(" ", "_")
                    surface = zone_srfs[sn]
                    srf_type = surface['surface_type']
                    cnstrct = construction[surface['construction_name']]
                    inner_layer, outer_layer, cthick = self.parse_cnstrct(cnstrct)
                    if self.mat_prims[inner_layer].primitive.identifier == '':
                        inner_layer = 'void'
                    else:
                        inner_layer = inner_layer.replace(' ','_')
                    srf_polygon = rg.Polygon([rg.Vector(*v.values())
                                              for v in surface['vertices']])
                    srf_windows = []
                    if sn in self.fene_hosts:
                        zone_fsrfs = {n:val for n, val in self.fene_srfs.items()
                                      if val['building_surface_name']==sn}
                        for fn in zone_fsrfs:
                            fsurface = zone_fsrfs[fn]
                            nfvert = int(fsurface['number_of_vertices'])
                            fverts = [[fsurface[k] for k in fsurface if
                                       k.startswith(f'vertex_{n+1}')] for n in range(nfvert)]
                            wndw_polygon = rg.Polygon([rg.Vector(*vert) for vert in fverts])
                            srf_windows.append(wndw_polygon)
                            window_construct = construction[fsurface['construction_name']]
                            window_material = self.parse_wndw_cnstrct(window_construct)
                            window_name = fn.replace(" ", "_")
                            zone['Window'][window_name] = ru.polygon2prim(
                                wndw_polygon, window_material, window_name)
                            srf_polygon -= wndw_polygon
                        facade = self.thicken(srf_polygon, srf_windows, cthick)
                        zone[srf_type][f'ext_{surface_name}'] = ru.polygon2prim(
                                facade[1], outer_layer, f"ext_{surface_name}")
                        for idx in range(2, len(facade)):
                            zone[srf_type][f'sill_{surface_name}.{idx}'] = ru.polygon2prim(
                                facade[idx], inner_layer, f"sill_{surface_name}.{idx}")
                    zone[srf_type][surface_name] = ru.polygon2prim(
                        srf_polygon, inner_layer, surface_name)
                ext_zones[zone_name] = self.check_srf_normal(zone)
        return ext_zones

    def get_site(self, epjs):
        site = epjs['Site:Location']
        for _, val in site.items():
            return val


def main():
    """Command-line program to convert a energyplus model into a Radiance model."""
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath')
    parser.add_argument('-run', action='store_true', default=False)
    args = parser.parse_args()
    epjs = read_epjs(args.fpath)
    radobj = epJSON2Rad(epjs)
    util.mkdir_p("Objects")
    util.mkdir_p("Resources")
    util.mkdir_p("Matrices")
    material_path = f"{util.basename(args.fpath)}_materials.mat"
    with open(os.path.join("Objects", material_path), 'w') as wtr:
        [wtr.write(str(val.primitive)) for val in radobj.mat_prims.values()]
    xml_paths = {}
    for key, val in radobj.matrices.items():
        opath = os.path.join('Resources', key+'.xml')
        tf_path = os.path.join('Resources', key+'_tf.mtx')
        tb_path = os.path.join('Resources', key+'_tb.mtx')
        with open(tf_path, 'w') as wtr:
            wtr.write(repr(val.tf))
        with open(tb_path, 'w') as wtr:
            wtr.write(repr(val.tb))
        basis = ''.join([word[0] for word in val.tf.basis.split()])
        cmd = ['wrapBSDF', '-f', 'n=%s'%key, '-a', basis]
        cmd += ['-tf', tf_path, '-tb', tb_path, '-U']
        wb_process = sp.run(cmd, check=True, stdout=sp.PIPE, stderr=sp.PIPE)
        with open(opath, 'wb') as wtr:
            wtr.write(wb_process.stdout)
        xml_paths[key] = opath
    for zn in radobj.zones:
        zone = radobj.zones[zn]
        scene_paths: List[str] = []
        window_paths: List[str] = []
        window_xml_paths: List[str] = []
        floors: List[str] = []
        for stype, surface in zone.items():
            if stype == 'Window':
                for key, val in surface.items():
                    window_path = f"{key}.rad"
                    _path = os.path.join("Objects", window_path)
                    window_paths.append(window_path)
                    with open(_path, 'w') as wtr:
                        wtr.write(str(val))
                    if val.modifier in radobj.matrices:
                        window_xml_paths.append(os.path.basename(xml_paths[val.modifier]))
            elif surface != {}:
                _name = f"{zn}_{stype}.rad".replace(" ", "_")
                _path = os.path.join("Objects", _name)
                with open(_path, 'w') as wtr:
                    for val in surface.values():
                        wtr.write(str(val))
                scene_paths.append(_name)
                if stype == "Floor":
                    floors.append(_name)

        file_struct = {'base': os.getcwd(), 'objects': "Objects",
                       'matrices': "Matrices", 'resources': "Resources",
                       'results': "Results"}
        model = {'material': material_path, 'scene': ' '.join(scene_paths),
                 'window_paths': ' '.join(window_paths),
                 'window_xml': ' '.join(window_xml_paths), 'window_cfs': '',
                 'window_control': ' '.join(map(str, range(len(window_xml_paths)))),
                 }
        if len(floors) > 1:
            logger.warning("More than one floor in this zone")
        floor0 = floors[0]
        raysender = {'grid_surface': floor0, 'grid_spacing': GRID_SPACING,
                     'grid_height': GRID_HEIGHT, 'view': ''}
        site = {'wea_path':'', 'latitude': radobj.site['latitude'],
                'longitude':radobj.site['longitude'], 'zipcode':''}
        templ_config = {"File Structure": file_struct, "Site": site,
                        "Model": model, "Ray Sender": raysender}
        if args.run:
            pass
            # mtxmtd = mm.MTXMethod(cfg)
        else:
            config = ConfigParser(allow_no_value=True)
            config.read_dict(templ_config)
            with open('%s.cfg'%zn, 'w') as wtr:
                config.write(wtr)
