"T.Wang"
from frads import radgeom
from frads import radutil
import pdb


class Room(object):
    """Make a shoebox."""

    def __init__(self, width, depth, height, origin=radgeom.Vector()):
        self.width = width
        self.depth = depth
        self.height = height
        self.origin = origin
        flr_pt2 = origin + radgeom.Vector(width, 0, 0)
        flr_pt3 = flr_pt2 + radgeom.Vector(0, depth, 0)
        self.floor = radgeom.Rectangle3P(origin, flr_pt2, flr_pt3)
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

        ceiling = {'modifier': 'white_paint_80', 'identifier': 'ceiling'}
        ceiling['real_args'] = self.clng.to_real()
        ceiling.update(_temp)
        self.srf_prims.append(ceiling)

        floor = {'modifier': 'carpet_25', 'identifier': 'floor'}
        floor['real_args'] = self.floor.to_real()
        floor.update(_temp)
        self.srf_prims.append(floor)

        nwall = {'modifier': 'white_paint_50', 'identifier': 'nwall'}
        nwall['real_args'] = self.nwall.polygon.to_real()
        nwall.update(_temp)
        self.srf_prims.append(nwall)

        ewall = {'modifier': 'white_paint_50', 'identifier': 'ewall'}
        ewall['real_args'] = self.ewall.polygon.to_real()
        ewall.update(_temp)
        self.srf_prims.append(ewall)

        wwall = {'modifier': 'white_paint_50', 'identifier': 'wwall'}
        wwall['real_args'] = self.wwall.polygon.to_real()
        wwall.update(_temp)
        self.srf_prims.append(wwall)

        # Windows on south wall only, for now.
        for idx in range(len(self.swall.facade)):
            _id = {'modifier': 'white_paint_50'}
            _id['identifier'] = 'swall_{:02d}'.format(idx)
            _id['real_args'] = self.swall.facade[idx].to_real()
            _id.update(_temp)
            self.srf_prims.append(_id)

    def window_prim(self):
        self.wndw_prims = {}
        for wpolygon in self.swall.windows:
            win_prim = {
                'modifier': 'glass_80',
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
        self.vect1 = (self.vertices[1] - self.vertices[0]).unitize()
        self.vect2 = (self.vertices[2] - self.vertices[1]).unitize()
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
            win_polygon = radgeom.Rectangle3P(win_pt3, win_pt1, win_pt2)
        return win_polygon

    def add_window(self, name, window_polygon):
        self.polygon = self.polygon - window_polygon
        self.windows[name] = window_polygon

    def facadize(self, thickness):
        direction = self.polygon.normal().scale(thickness)
        self.facade = self.polygon.extrude(direction)
        offset_wndw = {}
        for wndw in self.windows:
            offset_wndw[wndw] = radgeom.Polygon(
                [v + direction for v in self.windows[wndw].vertices])
        self.windows = offset_wndw


if __name__ == "__main__":
    rm1 = Room(10, 15, 11)
    swall = rm1.swall
    swin = swall.make_window(.5, 1, 9, 8.3)
    swall.add_window(swin)
    swall.facadize(.33)
    prims = rm1.to_prim()
    if rm1.window_prim():
        with open('window.rad', 'w') as wtr:
            wtr.write(radutil.put_primitive(rm1.win_prim))
    prim_str = [radutil.put_primitive(i) for i in list(prims.values())]
    with open('test.rad', 'w') as wtr:
        [wtr.write(i) for i in prim_str]
