"""Generic room model"""

import argparse
import os
from frads import radgeom
from frads import radutil


class Room(object):
    """Make a shoebox."""

    def __init__(self, width, depth, height, origin=radgeom.Vector()):
        self.width = width
        self.depth = depth
        self.height = height
        self.origin = origin
        flr_pt2 = origin + radgeom.Vector(width, 0, 0)
        flr_pt3 = flr_pt2 + radgeom.Vector(0, depth, 0)
        self.floor = radgeom.Polygon.rectangle3pts(origin, flr_pt2, flr_pt3)
        extrusion = self.floor.extrude(radgeom.Vector(0, 0, height))
        self.clng = extrusion[1]
        self.swall = Wall(extrusion[2], 'swall')
        self.ewall = Wall(extrusion[3], 'ewall')
        self.nwall = Wall(extrusion[4], 'nwall')
        self.wwall = Wall(extrusion[5], 'wwall')
        self.surfaces = [
            self.clng, self.floor, self.wwall, self.nwall, self.ewall,
            self.swall
        ]

    def surface_prim(self):
        self.srf_prims = []
        _temp = {'type': 'polygon', 'str_args': '0', 'int_arg': '0'}

        ceiling = {'modifier': 'white_paint_70', 'identifier': 'ceiling'}
        ceiling['real_args'] = self.clng.to_real()
        ceiling.update(_temp)
        self.srf_prims.append(ceiling)

        floor = {'modifier': 'carpet_20', 'identifier': 'floor'}
        floor['real_args'] = self.floor.to_real()
        floor.update(_temp)
        self.srf_prims.append(floor)

        nwall = {'modifier': 'white_paint_50', 'identifier': 'wall.north'}
        nwall['real_args'] = self.nwall.polygon.to_real()
        nwall.update(_temp)
        self.srf_prims.append(nwall)

        ewall = {'modifier': 'white_paint_50', 'identifier': 'wall.east'}
        ewall['real_args'] = self.ewall.polygon.to_real()
        ewall.update(_temp)
        self.srf_prims.append(ewall)

        wwall = {'modifier': 'white_paint_50', 'identifier': 'wall.west'}
        wwall['real_args'] = self.wwall.polygon.to_real()
        wwall.update(_temp)
        self.srf_prims.append(wwall)

        # Windows on south wall only, for now.
        for idx in range(len(self.swall.facade)):
            _id = {'modifier': 'white_paint_50'}
            _id['identifier'] = 'wall.south.{:02d}'.format(idx)
            _id['real_args'] = self.swall.facade[idx].to_real()
            _id.update(_temp)
            self.srf_prims.append(_id)

    def window_prim(self):
        self.wndw_prims = {}
        for wpolygon in self.swall.windows:
            win_prim = {
                'modifier': 'glass_60',
                'type': 'polygon',
                'str_args': '0',
                'int_arg': '0'
            }
            win_prim['identifier'] = wpolygon
            win_prim['real_args'] = self.swall.windows[wpolygon].to_real()
            win_prim['polygon'] = self.swall.windows[wpolygon]
            self.wndw_prims[wpolygon] = win_prim


class Wall(object):
    """Room wall object."""

    def __init__(self, polygon, name):
        self.centroid = polygon.centroid()
        self.polygon = polygon
        self.vertices = polygon.vertices
        self.vect1 = (self.vertices[1] - self.vertices[0]).normalize()
        self.vect2 = (self.vertices[2] - self.vertices[1]).normalize()
        self.name = name
        self.windows = {}

    def make_window(self, dist_left, dist_bot, width, height, wwr=None):
        if wwr is not None:
            assert type(wwr) == float, 'WWR must be float'
            win_polygon = self.polygon.scale(radgeom.Vector(*[wwr] * 3),
                                             self.centroid)
        else:
            win_pt1 = self.vertices[0]\
                    + self.vect1.scale(dist_bot)\
                    + self.vect2.scale(dist_left)
            win_pt2 = win_pt1 + self.vect1.scale(height)
            win_pt3 = win_pt1 + self.vect2.scale(width)
            win_polygon = radgeom.Polygon.rectangle3pts(win_pt3, win_pt1, win_pt2)
        return win_polygon

    def add_window(self, name, window_polygon):
        self.polygon = self.polygon - window_polygon
        self.windows[name] = window_polygon

    def facadize(self, thickness):
        direction = self.polygon.normal().scale(thickness)
        if thickness > 0:
            self.facade = self.polygon.extrude(direction)[:2]
            [self.facade.extend(self.windows[wname].extrude(direction)[2:])
             for wname in self.windows]
            uniq = []
            uniq = self.facade.copy()
            for idx in range(len(self.facade)):
                for re in self.facade[:idx]+self.facade[idx+1:]:
                    if set(self.facade[idx].to_list()) == set(re.to_list()):
                        uniq.remove(re)
            self.facade = uniq
        else:
            self.facade = [self.polygon]
        offset_wndw = {}
        for wndw in self.windows:
            offset_wndw[wndw] = radgeom.Polygon(
                [v + direction for v in self.windows[wndw].vertices])
        self.windows = offset_wndw

def make_room(dimension):
    """Make a side-lit shoebox room as a Room object."""
    theroom = Room(float(dimension['width']),
                      float(dimension['depth']),
                      float(dimension['height']))
    wndw_names = [i for i in dimension if i.startswith('window')]
    for wd in wndw_names:
        wdim = map(float, dimension[wd].split())
        theroom.swall.add_window(wd, theroom.swall.make_window(*wdim))
    theroom.swall.facadize(float(dimension['facade_thickness']))
    theroom.surface_prim()
    theroom.window_prim()
    return theroom

def genradroom():
    """Commandline interface for generating a generic room.
    Resulting Radiance .rad files will be written to a local
    Objects directory, which will be created if not existed before."""

    parser = argparse.ArgumentParser()
    parser.add_argument('width', type=float)
    parser.add_argument('depth', type=float)
    parser.add_argument('height', type=float)
    parser.add_argument('-w', '--window', nargs=4, action='append', type=float)
    parser.add_argument('-t', '--facade-thickness', type=float)
    args = parser.parse_args()
    dims = vars(args)
    for idx, window in enumerate(dims['window']):
        dims['window%s'%idx] = ' '.join(map(str, window))
    dims.pop('window')
    room = make_room(dims)
    material_primitives = radutil.material_lib()
    radutil.mkdir_p('Objects')
    with open(os.path.join('Objects', 'materials.mat'), 'w') as wtr:
        for prim in material_primitives:
            wtr.write(radutil.put_primitive(prim))
    with open(os.path.join('Objects', 'ceiling.rad'), 'w') as wtr:
        for prim in room.srf_prims:
            if prim['identifier'].startswith('ceiling'):
                wtr.write(radutil.put_primitive(prim))
    with open(os.path.join('Objects', 'floor.rad'), 'w') as wtr:
        for prim in room.srf_prims:
            if prim['identifier'].startswith('floor'):
                wtr.write(radutil.put_primitive(prim))
    with open(os.path.join('Objects', 'wall.rad'), 'w') as wtr:
        for prim in room.srf_prims:
            if prim['identifier'].startswith('wall'):
                wtr.write(radutil.put_primitive(prim))
    for key, prim in room.wndw_prims.items():
        with open(os.path.join('Objects', '%s.rad'%key), 'w') as wtr:
            wtr.write(radutil.put_primitive(prim))
