from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pyradiance as pr
from frads import utils
from frads.geom import Polygon
from pyradiance.lib import Primitive


@dataclass
class WindowSurface:
    polygon: Polygon
    primitive: pr.Primitive


@dataclass
class Surface:
    base: Polygon
    base_primitive: pr.Primitive
    polygons: List[Polygon]
    windows: List[WindowSurface]
    modifier: str
    identifier: str
    primitives: List[pr.Primitive]

    def move_window(self, distance: float) -> None:
        """Move windows in its normal direction."""
        direction = self.base.normal * distance
        new_windows = []
        for window in self.windows:
            new_polygon = window.polygon.move(direction)
            new_primitive = utils.polygon_primitive(
                polygon=new_polygon,
                modifier=window.primitive.modifier,
                identifier=window.primitive.identifier,
            )
            new_windows.append(WindowSurface(new_polygon, new_primitive))
        self.windows = new_windows

    def rotate_z(self, radians):
        """Rotate the surface counter clock-wise."""
        center = np.zeros(3)
        zaxis = np.array((0, 0, 1))
        new_base = self.base.rotate(center, zaxis, radians)
        new_base_primitive = utils.polygon_primitive(
            polygon=new_base,
            modifier=self.base_primitive.modifier,
            identifier=self.base_primitive.identifier,
        )
        new_polygons = [plg.rotate(center, zaxis, radians) for plg in self.polygons]
        new_primitives = []
        for idx, polygon in enumerate(new_polygons):
            new_primitives.append(
                utils.polygon_primitive(
                    polygon=polygon,
                    modifier=self.primitives[idx].modifier,
                    identifier=self.primitives[idx].identifier,
                )
            )
        new_windows = []
        for window in self.windows:
            new_polygon = window.polygon.rotate(center, zaxis, radians)
            new_primitive = utils.polygon_primitive(
                polygon=new_polygon,
                modifier=window.primitive.modifier,
                identifier=window.primitive.identifier,
            )
            new_windows.append(WindowSurface(new_polygon, new_primitive))

        self.polygons = new_polygons
        self.primitives = new_primitives
        self.windows = new_windows
        self.base = new_base
        self.base_primitive = new_base_primitive


@dataclass
class Room:
    floor: Surface
    ceiling: Surface
    swall: Surface
    ewall: Surface
    nwall: Surface
    wwall: Surface
    materials: List[Primitive]

    def primitives(self) -> List[pr.Primitive]:
        return [
            *self.floor.primitives,
            *self.ceiling.primitives,
            *self.swall.primitives,
            *self.ewall.primitives,
            *self.nwall.primitives,
            *self.wwall.primitives,
        ]

    def window_primitives(self) -> List[pr.Primitive]:
        return [
            *[srf.primitive for srf in self.ceiling.windows],
            *[srf.primitive for srf in self.swall.windows],
            *[srf.primitive for srf in self.ewall.windows],
            *[srf.primitive for srf in self.nwall.windows],
            *[srf.primitive for srf in self.wwall.windows],
        ]

    def model_dump(self) -> dict:
        model = {}
        model["materials"] = {"bytes": b" ".join(p.bytes for p in self.materials)}
        model["scene"] = {"bytes": b" ".join(p.bytes for p in self.primitives())}
        model["windows"] = {}
        for primitive in self.window_primitives():
            model["windows"][primitive.identifier] = {"bytes": primitive.bytes}
        model["surfaces"] = {
            "floor": {"primitives": [self.floor.base_primitive]},
            "ceiling": {"primitives": [self.ceiling.base_primitive]},
            "swall": {"primitives": [self.swall.base_primitive]},
            "ewall": {"primitives": [self.ewall.base_primitive]},
            "nwall": {"primitives": [self.nwall.base_primitive]},
            "wwall": {"primitives": [self.wwall.base_primitive]},
        }
        return model

    def rotate_z(self, radians):
        """Rotate the room counter clock-wise."""
        self.floor.rotate_z(radians)
        self.ceiling.rotate_z(radians)
        self.swall.rotate_z(radians)
        self.ewall.rotate_z(radians)
        self.wwall.rotate_z(radians)
        self.nwall.rotate_z(radians)

    def validate(self) -> None:
        """Validate the room model."""
        material_names = [p.identifier for p in self.materials]
        for prim in self.primitives():
            if prim.modifier not in material_names:
                raise ValueError(
                    f"Unknown modifier {prim.modifier} in {prim.identifier}"
                )


def thicken(base, thickness) -> List[Polygon]:
    """Thicken the surface."""
    direction = base.normal * thickness
    polygons = base.extrude(direction)
    # Remove duplicates.
    counts = [polygons.count(plg) for plg in polygons]
    polygons = [plg for plg, cnt in zip(polygons, counts) if cnt == 1]
    return polygons


def make_window(
    name: str,
    base: Polygon,
    vertices: np.ndarray,
    upvec: np.ndarray,
    rightvec: np.ndarray,
    dist_left: float,
    dist_bot: float,
    width: float,
    height: float,
) -> Tuple[Polygon, WindowSurface]:
    """Make one or more window and punch a hole.

    -----------------
    |      __       |
    |     |  |      |
    |__a__|__|      |
    |     |         |
    |     |b        |
    -----------------

    a: dist_left
    b: dist_bot

    Args:
        base: The base polygon.
        vertices: The vertices of the original base polygon.
        vec1: The unit vector of the first edge of the original base polygon.
        vec2: The unit vector of the second edge of the original base polygon.
        dist_left: The distance from the left edge of the base polygon.
        dist_bot: The distance from the bottom edge of the base polygon.
        width: The width of the window.
        height: The height of the window.
    """
    win_pt1 = vertices[0] + upvec * dist_bot + rightvec * dist_left
    win_pt2 = win_pt1 + upvec * height
    win_pt3 = win_pt1 + rightvec * width
    window_polygon = Polygon.rectangle3pts(win_pt3, win_pt1, win_pt2)
    new_base = base - window_polygon
    return new_base, WindowSurface(
        polygon=window_polygon,
        primitive=utils.polygon_primitive(
            polygon=window_polygon,
            modifier="glass_60",
            identifier=name,
        ),
    )


def make_window_wwr(base: Polygon, wwr: float) -> Tuple[Polygon, WindowSurface]:
    """
    Make a single window and punch a hole based on window-to-wall ratio.

    """
    window_polygon = base.scale(np.array((wwr, wwr, wwr)), base.centroid)
    base = base - window_polygon
    return base, WindowSurface(
        polygon=window_polygon,
        primitive=utils.polygon_primitive(
            polygon=window_polygon,
            modifier="glass_60",
            identifier="void",
        ),
    )


def create_surface(
    base: Polygon,
    thickness: float = 0,
    modifier: str = "void",
    identifier: str = "void",
    wpd: Optional[List[List[float]]] = None,
    wwr: Optional[float] = None,
) -> Surface:
    """Create a surface with windows.

    Args:
        base: The base polygon.
        thickness: The thickness of the surface.
        modifier: The modifier of the surface.
        identifier: The identifier of the surface.
        wpd: The window position and dimension, mutex to wwr.
        wwr: The window-to-wall ratio, mutex to wpd.

    Returns:
        A Surface object.
    """
    vertices = base.vertices
    polygons = [base]
    windows = []
    base_primitive = utils.polygon_primitive(
        polygon=base,
        modifier=modifier,
        identifier=identifier,
    )
    primitives = [
        utils.polygon_primitive(
            polygon=base,
            modifier=modifier,
            identifier=identifier,
        )
    ]
    if wpd is not None:
        # Make a window based on window position and dimension.
        for i, pd in enumerate(wpd):
            name = f"{identifier}_window{i}"
            # vec1 = (vertices[1] - vertices[0]) / np.linalg.norm(vertices[1] - vertices[0])
            # vec2 = (vertices[2] - vertices[1]) / np.linalg.norm(vertices[2] - vertices[1])
            upvec = np.array((0, 0, 1))
            rightvec = np.array((1, 0, 0))
            base, _window = make_window(name, base, vertices, upvec, rightvec, *pd)
            windows.append(_window)
    elif wwr is not None:
        # Make a window based on window-to-wall ratio.
        base, _window = make_window_wwr(base, wwr)
        windows.append(_window)
    if thickness > 0:
        polygons = thicken(base, thickness)
        primitives = [
            utils.polygon_primitive(
                polygon=polygon,
                modifier=modifier,
                identifier=f"{identifier}_{i}",
            )
            for i, polygon in enumerate(polygons)
        ]
    return Surface(
        base,
        base_primitive,
        polygons,
        windows,
        modifier,
        identifier,
        primitives,
    )


def create_south_facing_room(
    width: float,
    depth: float,
    floor_floor: float,
    floor_ceiling: float,
    swall_thickness: float = 0,
    wpd: Optional[List[List[float]]] = None,
    wwr: Optional[float] = None,
) -> Room:
    materials = list(utils.material_lib().values())
    pt1 = np.array((0, 0, 0))
    pt2 = pt1 + np.array((0, depth, 0))
    pt3 = pt2 + np.array((width, 0, 0))
    base_floor = Polygon.rectangle3pts(pt1, pt2, pt3)
    # _, base_ceiling, base_wwall, base_nwall, base_ewall, base_swall = (
    #     base_floor.extrude(np.array((0, 0, floor_floor)))
    # )
    base_ceiling = base_floor.flip().move(np.array((0, 0, floor_ceiling)))
    base_nwall = Polygon(
        [
            np.array((width, depth, 0)),
            np.array((0, depth, 0)),
            np.array((0, depth, floor_floor)),
            np.array((width, depth, floor_floor)),
        ]
    )
    base_swall = Polygon(
        [
            np.array((0, 0, 0)),
            np.array((width, 0, 0)),
            np.array((width, 0, floor_floor)),
            np.array((0, 0, floor_floor)),
        ]
    )
    base_ewall = Polygon(
        [
            np.array((width, 0, 0)),
            np.array((width, depth, 0)),
            np.array((width, depth, floor_floor)),
            np.array((width, 0, floor_floor)),
        ]
    )
    base_wwall = Polygon(
        [
            np.array((0, 0, 0)),
            np.array((0, 0, floor_floor)),
            np.array((0, depth, floor_floor)),
            np.array((0, depth, 0)),
        ]
    )
    floor = create_surface(
        base_floor,
        modifier="neutral_lambertian_0.2",
        identifier="floor",
    )
    ceiling = create_surface(
        base_ceiling,
        modifier="neutral_lambertian_0.7",
        identifier="ceiling",
    )
    nwall = create_surface(
        base_nwall,
        modifier="neutral_lambertian_0.5",
        identifier="nwall",
    )
    ewall = create_surface(
        base_ewall,
        modifier="neutral_lambertian_0.5",
        identifier="ewall",
    )
    wwall = create_surface(
        base_wwall,
        modifier="neutral_lambertian_0.5",
        identifier="wwall",
    )
    swall = create_surface(
        base_swall,
        thickness=swall_thickness,
        modifier="neutral_lambertian_0.5",
        identifier="swall",
        wpd=wpd,
        wwr=wwr,
    )
    return Room(
        floor,
        ceiling,
        swall,
        ewall,
        nwall,
        wwall,
        materials,
    )
