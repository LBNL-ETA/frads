import json
from frads import radutil as ru
from frads import radgeom as rg
import os
import pdb
import logging

logger = logging.getLogger("frads.epjson2rad")


class epJSON2Rad(object):

    # Sensor grid dimension in meters
    GRID_HEIGHT = 0.75
    GRID_SPACING = 0.6

    def __init__(self, epjs):
        self.checkout_fene(epjs)
        self.get_material_prims(epjs)
        self.get_zones(epjs)

    def get_thickness(self, layers):
        """Get thickness from construction."""
        thickness = 0
        for l in layers:
            try:
                thickness += self.mat_prims[l]['thickness']
            except KeyError:
                thickness += 0
        return thickness

    def thicken(self, surface, windows, thickness):
        """Thicken window-wall."""
        direction = surface.normal().scale(thickness)
        facade = surface.extrude(direction)[:2]
        [facade.extend(windows[wname].extrude(direction)[2:]) for wname in windows]
        uniq = facade.copy()
        for idx in range(len(facade)):
            for re in facade[:idx]+facade[idx+1:]:
                if set(facade[idx].to_list()) == set(re.to_list()):
                    uniq.remove(re)
        return uniq


    def get_material(self, epjs_mat):
        mat_prims = {}
        for mat in epjs_mat:
            mat_prims[mat] = {'modifier':'void', 'int_arg':'0',
                              'str_args':'0', 'type':'plastic',
                              'identifier':mat.replace(' ','_')}
            try:
                refl = 1 - epjs_mat[mat]['visible_absorptance']
            except KeyError as ke:
                logger.warning(f"No visible absorptance defined for {mat}, assuming 50%")
                refl = .5
            mat_prims[mat]['real_args'] = "5 {0:.2f} {0:.2f} {0:.2f} 0 0".format(refl)
            try:
                mat_prims[mat]['thickness'] = epjs_mat[mat]['thickness']
            except KeyError as ke:
                logger.info(f"{mat} has zero thickness")
                mat_prims[mat]['thickness'] = 0
        return mat_prims

    def get_wndw_material(self, epjs_wndw_mat):
        wndw_mat_prims = {}
        for mat in epjs_wndw_mat:
            try:
                tvis = epjs_wndw_mat[mat]['visible_transmittance']
            except KeyError:
                print(f"No visible transmittance defined for {mat}, assuming 60%")
                tvis = 0.6
            wndw_mat_prims[mat] = {'modifier':'void', 'type':'glass', 'int_arg':'0',
                              'str_args':'0', 'identifier':mat.replace(' ','_'),
                              'real_args': "3 {0:.2f} {0:.2f} {0:.2f}".format(tvis)}
        return wndw_mat_prims

    def get_material_prims(self, epjs):
        self.mat_prims = {}
        try:
            self.mat_prims.update(self.get_material(epjs['Material']))
        except KeyError as ke:
            logger.info(ke, ", moving on")
        try:
            self.mat_prims.update(self.get_material(epjs['Material:NoMass']))
        except KeyError as ke:
            logger.info(ke, ", moving on")
        try:
            self.mat_prims.update(self.get_wndw_material(epjs['WindowMaterial:SimpleGalzingSystem']))
        except KeyError as ke:
            logger.info(ke, ", moving on")

    def checkout_fene(self, epjs):
        try:
            self.fene_srfs = epjs['FenestrationSurface:Detailed']
        except KeyError as ke:
            raise Exception(ke, 'not found, no exterior window found.')
        self.fene_hosts = set([self.fene_srfs[name]['building_surface_name']
                               for name in self.fene_srfs])

    def check_ext_window(self, zone_srfs):
        has_window = False
        for sn in zone_srfs:
            if (sn in self.fene_hosts) and (zone_srfs[sn]['sun_exposure']!='NoSun'):
                has_window = True
                break
        return has_window

    def parse_cnstrct(self, cnstrct):
        layers = ['outside_layer']
        layers.extend(sorted([l for l in cnstrct if l.startswith('layer_')]))
        inner_layer = cnstrct[layers[-1]].replace(' ','_')
        outer_layer = cnstrct[layers[0]].replace(' ','_')
        cthick = sum([self.mat_prims[cnstrct[l]]['thickness'] for l in layers])
        return inner_layer, outer_layer, cthick

    def parse_wndw_cnstrct(self, wcnstrct):
        layers = ['outside_layer']
        layers.extend(sorted([l for l in wcnstrct if l.startswith('layer_')]))
        pass

    def get_zones(self, epjs):
        opaque_srfs = epjs['BuildingSurface:Detailed']
        ext_zones = {}
        for zn in epjs['Zone']:
            ext_zones[zn] = {'Wall':{}, 'Floor':{}, 'Ceiling':{}, 'Window':{}}
            zone_srfs = {name:opaque_srfs[name] for name in opaque_srfs
                        if opaque_srfs[name]['zone_name'] == zn}
            if self.check_ext_window(zone_srfs):
                wsrf_prims = []
                for sn in zone_srfs:
                    pdb.set_trace()
                    surface = zone_srfs[sn]
                    srf_type = surface['surface_type']
                    cnstrct = epjs['Construction'][surface['construction_name']]
                    inner_layer, outer_layer, cthick = self.parse_cnstrct(cnstrct)
                    srf_polygon = rg.Polygon([rg.Vector(*v.values())
                                              for v in surface['vertices']])
                    srf_windows = {}
                    if sn in self.fene_hosts:
                        zone_fsrfs = {name:self.fene_srfs[name] for name in self.fene_srfs
                                      if self.fene_srfs[name]['building_surface_name']==sn}
                        for fn in zone_fsrfs:
                            fsurface = zone_fsrfs[fn]
                            nfvert = fsurface['number_of_vertices']
                            fverts = [[fsurface[k] for k in fsurface if
                                       k.startswith(f'vertex_{n+1}')] for n in range(nfvert)]
                            wndw_polygon = rg.Polygon([rg.Vector(*vert) for vert in fverts])
                            wcnstrct = epjs['Construction'][fsurface['construction_name']]
                            # self.parse_wndw_cnstrct(wcnstrct)
                            wndw_mat = wcnstrct['outside_layer'].replace(' ','_')
                            ext_zones[zn]['Window'][fn] = ru.polygon2prim(wndw_polygon, wndw_mat, fn)
                            srf_polygon -= wndw_polygon
                        facade = self.thicken(srf_polygon, srf_windows, cthick)
                        ext_zones[zn][srf_type][f'ext_{sn}'] = ru.polygon2prim(
                                facade[1], outer_layer, f"ext_{sn}")
                        for idx in range(2, len(facade)):
                            ext_zones[zn][srf_type][f'sill_{sn}.{idx}'] = ru.polygon2prim(
                                facade[idx], inner_layer, f"sill_{sn}.{idx}")
                    ext_zones[zn][srf_type][sn] = ru.polygon2prim(srf_polygon, inner_layer, sn)
        return ext_zones


    def check_normal(self, window, surfaces):
        """Check the surface normal orientation
        Arguments:
            Window: a polygon representing the window
            floor: a polygon representing the floor"""
        win_norm = window.normal()
        win_center = window.center()
        pcentroid = radgeom.polygon_center(surfaces)


def read_epjs(fpath):
    with open(fpath) as rdr:
        epjs = json.load(rdr)
    return epjs






