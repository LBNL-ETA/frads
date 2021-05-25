"""
Convert an EnergyPlus model as a parsed dictionary
to Radiance primitives.
Todo:
    * Parse window data file for Construction:WindowDataFile

"""
from collections import namedtuple
import logging
from frads import radutil as ru
from frads import radgeom as rg
from frads import util

logger = logging.getLogger("frads.epjson2rad")

PI = 3.14159265358579
Primitive = ru.Primitive

class epJSON2Rad:
    """
    Convert EnergyPlus JSON objects into Radiance primitives.
    """
    material = namedtuple("material", ["primitive", "thickness"])

    def __init__(self, epjs):
        """Parse EPJSON object passed in as a dictionary."""
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
                refl = 1 - val['visible_absorptance']
                _identifier = key.replace(' ','_')
                _real_arg = "5 {0:.2f} {0:.2f} {0:.2f} 0 0".format(refl)
                _thickness = 0
            mat_prims[key] = self.material(
                Primitive(_modifier, _type, _identifier, '0', _real_arg),
                0) # zero thickness
        return mat_prims

    def _windowmaterial_gap(self, epjs_wndw_mat):
        return {}

    def _windowmaterial_gas(self, epjs_wndw_mat):
        return {}


    def _windowmaterial_simpleglazingsystem(self, epjs_wndw_mat):
        """Parse EP WindowMaterial:Simpleglazing"""
        wndw_mat_prims = {}
        for key, val in epjs_wndw_mat.items():
            try:
                tmis = util.tmit2tmis(val['visible_transmittance'])
            except KeyError as ke:
                raise Exception(ke, "for", key)
            wndw_mat_prims[key] = Primitive(
                'void', 'glass', key.replace(' ','_'), '0',
                "3 {0:.2f} {0:.2f} {0:.2f}".format(tmis))
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
            wndw_mat_prims[key] = Primitive(
                'void', 'glass', key.replace(' ','_'), '0',
                "3 {0:.2f} {0:.2f} {0:.2f}".format(tmis))
        return wndw_mat_prims

    def _windowmaterial_glazing(self, epjs_wndw_mat):
        """Parse EP WindowMaterial:Glazing"""
        wndw_mat_prims = {}
        for key, val in epjs_wndw_mat.items():
            if val['optical_data_type'].lower() == 'bsdf':
                wndw_mat_prims[key] = None
            else:
                tvis = val['visible_transmittance_at_normal_incidence']
                tmis = util.tmit2tmis(tvis)
                wndw_mat_prims[key] = Primitive(
                    'void', 'glass', key.replace(' ','_'), '0',
                    "3 {0:.2f} {0:.2f} {0:.2f}".format(tmis))
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
        layers = ['outside_layer']
        layers.extend(sorted([l for l in cnstrct if l.startswith('layer_')]))
        inner_layer = cnstrct[layers[-1]]
        outer_layer = cnstrct[layers[0]].replace(' ','_')
        cthick = sum([self.mat_prims[cnstrct[l]].thickness for l in layers])
        return inner_layer, outer_layer, cthick

    def _construction_windowdatafile(self, construction):
        file_path = construction['file_name']
        # self.parse_wndw_data(file_path)
        # return layers, cthick

    def parse_wndw_cnstrct(self, wcnstrct):
        """Parse window construction."""
        layers = ['outside_layer']
        layers.extend(sorted([l for l in wcnstrct if l.startswith('layer_')]))

    def check_srf_normal(self, zone):
        """Check the surface normal in each zone."""
        polygons = [v['polygon'] for k,v in zone['Floor'].items()]
        polygons.extend([v['polygon'] for k,v in zone['Ceiling'].items()])
        polygons.extend([v['polygon'] for k,v in zone['Wall'].items()])
        polygons.extend([v['polygon'] for k,v in zone['Window'].items()])
        centroid = rg.polygon_center(*polygons)
        for k,v in zone['Floor'].items():
            fpolygon = v['polygon']
            angle2center = fpolygon.normal().angle_from(centroid - fpolygon.centroid())
            if angle2center < PI/4:
                v['polygon'] = fpolygon.flip()
                v['real_args'] = v['polygon'].to_real()
        for k,v in zone['Window'].items():
            wpolygon = v['polygon']
            angle2center = wpolygon.normal().angle_from(centroid - wpolygon.centroid())
            if angle2center > PI/4:
                v['polygon'] = wpolygon.flip()
                v['real_args'] = v['polygon'].to_real()
        return zone

    def get_construction(self, epjs):
        construction_key = [key for key in epjs if key.split(':')[0] == 'Construction']
        construction_dict = {}
        for key in construction_key:
            construction_dict.update(epjs[key])
        return construction_dict


    def get_zones(self, epjs):
        """Looping through zones."""
        opaque_srfs = epjs['BuildingSurface:Detailed']
        construction = self.get_construction(epjs)
        ext_zones = {}
        for zn in epjs['Zone']:
            zone_srfs = {k:v for k,v in opaque_srfs.items() if v['zone_name'] == zn}
            if self.check_ext_window(zone_srfs):
                zone = {'Wall':{}, 'Floor':{}, 'Ceiling':{}, 'Window':{}, 'Roof':{}}
                wsrf_prims = []
                for sn in zone_srfs:
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
                            wcnstrct = construction[fsurface['construction_name']]
                            # self.parse_wndw_cnstrct(wcnstrct)
                            wndw_mat = wcnstrct['outside_layer'].replace(' ','_')
                            zone['Window'][fn] = ru.polygon2prim(wndw_polygon, wndw_mat, fn)
                            srf_polygon -= wndw_polygon
                        facade = self.thicken(srf_polygon, srf_windows, cthick)
                        zone[srf_type][f'ext_{sn}'] = ru.polygon2prim(
                                facade[1], outer_layer, f"ext_{sn}")
                        for idx in range(2, len(facade)):
                            zone[srf_type][f'sill_{sn}.{idx}'] = ru.polygon2prim(
                                facade[idx], inner_layer, f"sill_{sn}.{idx}")
                    zone[srf_type][sn] = ru.polygon2prim(srf_polygon, inner_layer, sn)
                ext_zones[zn] = self.check_srf_normal(zone)
        return ext_zones

    def get_site(self, epjs):
        site = epjs['Site:Location']
        for _, val in site.items():
            return val
