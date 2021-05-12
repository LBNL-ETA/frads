"""Generic room model"""

import argparse
import os
from frads import radgeom
from frads import radutil, util


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
        self.wall_south = Surface(extrusion[2], 'wall.south')
        self.wall_east = Surface(extrusion[3], 'wall.east')
        self.wall_north = Surface(extrusion[4], 'wall.north')
        self.wall_west = Surface(extrusion[5], 'wall.west')
        self.surfaces = [
            self.clng, self.floor, self.wall_west, self.wall_north, self.wall_east,
            self.wall_south
        ]

    def surface_prim(self):
        self.srf_prims = []
        ceiling = radutil.Primitive(
            'white_paint_70', 'polygon', 'ceiling', '0', self.clng.to_real())
        self.srf_prims.append(ceiling)

        floor = radutil.Primitive(
            'carpet_20', 'polygon', 'floor', '0', self.floor.to_real())
        self.srf_prims.append(floor)

        nwall = radutil.Primitive(
            'white_paint_50', 'polygon', self.wall_north.name,
            '0', self.wall_north.polygon.to_real())
        self.srf_prims.append(nwall)

        ewall = radutil.Primitive('white_paint_50', 'polygon', self.wall_east.name,
                                  '0', self.wall_east.polygon.to_real())
        self.srf_prims.append(ewall)

        wwall = radutil.Primitive('white_paint_50', 'polygon', self.wall_west.name,
                                  '0', self.wall_west.polygon.to_real())
        self.srf_prims.append(wwall)

        # Windows on south wall only, for now.
        for idx, swall in enumerate(self.wall_south.facade):
            _identifier = '{}.{:02d}'.format(self.wall_south.name, idx)
            _id = radutil.Primitive(
                'white_paint_50', 'polygon', _identifier, '0', swall.to_real())
            self.srf_prims.append(_id)

    def window_prim(self):
        self.wndw_prims = {}
        for wpolygon in self.wall_south.windows:
            _real_args = self.wall_south.windows[wpolygon].to_real()
            win_prim = radutil.Primitive('glass_60', 'polygon', wpolygon, '0', _real_args)
            self.wndw_prims[wpolygon] = win_prim


class Surface(object):
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

def make_room(dimension: dict):
    """Make a side-lit shoebox room as a Room object."""
    theroom = Room(float(dimension['width']),
                   float(dimension['depth']),
                   float(dimension['height']))
    wndw_names = [i for i in dimension if i.startswith('window')]
    for wd in wndw_names:
        wdim = map(float, dimension[wd].split())
        theroom.wall_south.add_window(wd, theroom.wall_south.make_window(*wdim))
    theroom.wall_south.facadize(float(dimension['facade_thickness']))
    theroom.surface_prim()
    theroom.window_prim()
    return theroom


def genradroom():
    """Commandline interface for generating a generic room.
    Resulting Radiance .rad files will be written to a local
    Objects directory, which will be created if not existed before."""

    parser = argparse.ArgumentParser(
        prog='genradroom', description='Generate a generic room')
    parser.add_argument('width', type=float,
                        help='room width along X axis, starting from x=0')
    parser.add_argument('depth', type=float,
                        help='room depth along Y axis, starting from y=0')
    parser.add_argument('height', type=float,
                        help='room height along Z axis, starting from z=0')
    parser.add_argument('-w', dest='window',
                        metavar=('start_x', 'start_z', 'width', 'height'),
                        nargs=4, action='append', type=float,
                        help='Define a window from lower left corner')
    parser.add_argument('-n', dest='name', help='Model name', default='model')
    parser.add_argument('-t', dest='facade_thickness',
                        metavar='Facade thickness', type=float)
    args = parser.parse_args()
    dims = vars(args)
    for idx, window in enumerate(dims['window']):
        dims['window_%s' % idx] = ' '.join(map(str, window))
    dims.pop('window')
    room = make_room(dims)
    name = args.name
    material_primitives = radutil.material_lib()
    util.mkdir_p('Objects')
    with open(os.path.join('Objects', f'materials_{name}.mat'), 'w') as wtr:
        for prim in material_primitives:
            wtr.write(str(prim)+'\n')
    with open(os.path.join('Objects', f'ceiling_{name}.rad'), 'w') as wtr:
        for prim in room.srf_prims:
            if prim.identifier.startswith('ceiling'):
                wtr.write(str(prim)+'\n')
    with open(os.path.join('Objects', f'floor_{name}.rad'), 'w') as wtr:
        for prim in room.srf_prims:
            if prim.identifier.startswith('floor'):
                wtr.write(str(prim)+'\n')
    with open(os.path.join('Objects', f'wall_{name}.rad'), 'w') as wtr:
        for prim in room.srf_prims:
            if prim.identifier.startswith('wall'):
                wtr.write(str(prim)+'\n')
    for key, prim in room.wndw_prims.items():
        with open(os.path.join('Objects', f'{key}_{name}.rad'), 'w') as wtr:
            wtr.write(str(prim)+'\n')
