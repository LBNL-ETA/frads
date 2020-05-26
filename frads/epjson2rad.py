import json
from frads import radutil as ru
from frads import radgeom as rg
import os
import pdb
import logging

logger = logging.getLogger("frads.epjson2rad")

PI = 3.14159265358579

class epJSON2Rad(object):


    def __init__(self, epjs):
        self.checkout_fene(epjs)
        self.get_material_prims(epjs)
        self.zones = self.get_zones(epjs)

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
        [facade.extend(window.extrude(direction)[2:]) for window in windows]
        uniq = facade.copy()
        for idx in range(len(facade)):
            for re in facade[:idx]+facade[idx+1:]:
                if set(facade[idx].to_list()) == set(re.to_list()):
                    uniq.remove(re)
        return uniq


    def get_material(self, epjs_mat):
        mat_prims = {}
        for key, val in epjs_mat.items():
            mat_prims[key] = {'modifier':'void', 'int_arg':'0',
                              'str_args':'0', 'type':'plastic',
                              'identifier':key.replace(' ','_')}
            try:
                refl = 1 - val['visible_absorptance']
            except KeyError as ke:
                logger.warning(f"No visible absorptance defined for {key}, assuming 50%")
                refl = .5
            mat_prims[key]['real_args'] = "5 {0:.2f} {0:.2f} {0:.2f} 0 0".format(refl)
            try:
                mat_prims[key]['thickness'] = val['thickness']
            except KeyError as ke:
                logger.info(f"{key} has zero thickness")
                mat_prims[key]['thickness'] = 0
        return mat_prims

    def get_wndw_material(self, epjs_wndw_mat):
        wndw_mat_prims = {}
        for key, val in epjs_wndw_mat.items():
            try:
                tvis = val['visible_transmittance']
            except KeyError:
                print(f"No visible transmittance defined for {key}, assuming 60%")
                tvis = 0.6
            wndw_mat_prims[key] = {'modifier':'void', 'type':'glass', 'int_arg':'0',
                              'str_args':'0', 'identifier':key.replace(' ','_'),
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
            self.mat_prims.update(self.get_wndw_material(epjs['WindowMaterial:SimpleGlazingSystem']))
        except KeyError as ke:
            logger.info(ke, ", moving on")

    def checkout_fene(self, epjs):
        try:
            self.fene_srfs = epjs['FenestrationSurface:Detailed']
        except KeyError as ke:
            raise Exception(ke, 'not found, no exterior window found.')
        self.fene_hosts = set([val['building_surface_name']
                               for key, val in self.fene_srfs.items()])

    def check_ext_window(self, zone_srfs):
        has_window = False
        for key, val in zone_srfs.items():
            if (key in self.fene_hosts) and (val['sun_exposure']!='NoSun'):
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

    def check_srf_normal(self, zone):
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


    def get_zones(self, epjs):
        """Looping through zones."""
        opaque_srfs = epjs['BuildingSurface:Detailed']
        ext_zones = {}
        for zn in epjs['Zone']:
            zone_srfs = {k:v for k,v in opaque_srfs.items() if v['zone_name'] == zn}
            if self.check_ext_window(zone_srfs):
                zone = {'Wall':{}, 'Floor':{}, 'Ceiling':{}, 'Window':{}}
                wsrf_prims = []
                for sn in zone_srfs:
                    surface = zone_srfs[sn]
                    srf_type = surface['surface_type']
                    cnstrct = epjs['Construction'][surface['construction_name']]
                    inner_layer, outer_layer, cthick = self.parse_cnstrct(cnstrct)
                    srf_polygon = rg.Polygon([rg.Vector(*v.values())
                                              for v in surface['vertices']])
                    srf_windows = []
                    if sn in self.fene_hosts:
                        zone_fsrfs = {n:val for n, val in self.fene_srfs.items()
                                      if val['building_surface_name']==sn}
                        for fn in zone_fsrfs:
                            fsurface = zone_fsrfs[fn]
                            nfvert = fsurface['number_of_vertices']
                            fverts = [[fsurface[k] for k in fsurface if
                                       k.startswith(f'vertex_{n+1}')] for n in range(nfvert)]
                            wndw_polygon = rg.Polygon([rg.Vector(*vert) for vert in fverts])
                            srf_windows.append(wndw_polygon)
                            wcnstrct = epjs['Construction'][fsurface['construction_name']]
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


