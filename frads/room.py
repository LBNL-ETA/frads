from dataclasses import dataclass
from typing import List, Optional, Tuple

import pyradiance as pr
from pyradiance.lib import Primitive
from frads.geom import Polygon
from frads import utils
import numpy as np

@dataclass
class WindowSurface:
    polygon: Polygon
    primitive: pr.Primitive


@dataclass
class Surface2:
    base: Polygon
    base_primitive: pr.Primitive
    polygons: List[Polygon]
    windows: List[WindowSurface]
    modifier: str
    identifier: str
    primitives: List[pr.Primitive]


class Surface:
    """Surface object."""

    def __init__(self, base: Polygon, thickness:float =0) -> None:
        """."""
        self.base = base
        self.thickness = thickness
        self._vertices = base.vertices
        self._vect1 = (base.vertices[1] - base.vertices[0]) / np.linalg.norm(
            base.vertices[1] - base.vertices[0]
        )
        self._vect2 = (base.vertices[2] - base.vertices[1]) / np.linalg.norm(
            base.vertices[2] - base.vertices[1]
        )
        self.polygons: List[Polygon] = [self.base]
        self.windows: List[Surface] = []
        self._modifier: str = "void"
        self._identifier: str = "void"
        self._primitives: List[pr.Primitive] = [
            utils.polygon_primitive(
                polygon=self.base,
                modifier=self._modifier,
                identifier=self._identifier,
            )
        ]

    @property
    def modifier(self):
        """."""
        return self._modifier

    @property
    def identifier(self):
        """."""
        return self._identifier

    @modifier.setter
    def modifier(self, mod):
        """."""
        self._modifier = mod

    @identifier.setter
    def identifier(self, identifier):
        """."""
        self._identifier = identifier

    @property
    def primitives(self) -> List[pr.Primitive]:
        """."""
        for idx, polygon in enumerate(self.polygons):
            self._primitives.append(
                pr.Primitive(
                    self.modifier,
                    "polygon",
                    f"{self.identifier}_{idx:02d}",
                    [],
                    polygon.coordinates,
                )
            )
        return self._primitives

    def make_window_wwr(self, wwr: float) -> None:
        """Make a window based on window-to-wall ratio."""
        window_polygon = self.base.scale(np.array((wwr, wwr, wwr)), self.base.centroid)
        self.base = self.base - window_polygon
        self.windows.append(Surface(window_polygon))

    def make_window(
        self, dist_left: float, dist_bot: float, width: float, height: float
    ) -> None:
        """Make a window and punch a hole."""
        win_pt1 = self._vertices[0] + self._vect1 * dist_bot + self._vect2 * dist_left
        win_pt2 = win_pt1 + self._vect1 * height
        win_pt3 = win_pt1 + self._vect2 * width
        window_polygon = Polygon.rectangle3pts(win_pt3, win_pt1, win_pt2)
        self.base = self.base - window_polygon
        self.windows.append(Surface(window_polygon))

    def thicken(self) -> None:
        """Thicken the surface."""
        direction = self.base.normal * self.thickness
        polygons = self.base.extrude(direction)
        counts = [polygons.count(plg) for plg in polygons]
        self.polygons = [plg for plg, cnt in zip(polygons, counts) if cnt == 1]

    def move_window(self, distance: float) -> None:
        """Move windows in its normal direction."""
        direction = self.base.normal * distance
        self.windows = [Surface(window.base.move(direction)) for window in self.windows]

    def rotate(self, deg):
        """Rotate the surface counter clock-wise."""
        polygons = []
        center = np.zeros(3)
        zaxis = np.array((0, 0, 1))
        for plg in self.polygons:
            polygons.append(plg.rotate(center, zaxis, deg))
        self.polygons = polygons
        for window in self.windows:
            wpolygons = []
            for plg in window.polygons:
                wpolygons.append(plg.rotate(center, zaxis, deg))


@dataclass
class Room2:
    floor: Surface2
    ceiling: Surface2
    swall: Surface2
    ewall: Surface2
    nwall: Surface2
    wwall: Surface2
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
            "floor": {"bytes": self.floor.base_primitive.bytes},
            "ceiling": {"bytes": self.ceiling.base_primitive.bytes},
            "swall": {"bytes": self.swall.base_primitive.bytes},
            "ewall": {"bytes": self.ewall.base_primitive.bytes},
            "nwall": {"bytes": self.nwall.base_primitive.bytes},
            "wwall": {"bytes": self.wwall.base_primitive.bytes},
        }
        return model


class Room:
    """Make a shoebox."""

    def __init__(
        self,
        floor: Surface,
        ceiling: Surface,
        swall: Surface,
        ewall: Surface,
        nwall: Surface,
        wwall: Surface,
    ) -> None:
        """."""
        self.floor = floor
        self.ceiling = ceiling
        self.swall = swall
        self.ewall = ewall
        self.nwall = nwall
        self.wwall = wwall
        self.materials = utils.material_lib()

    @classmethod
    def from_wdh(
        cls,
        width: float,
        depth: float,
        floor_floor: float,
        floor_ceiling: float,
        origin: Optional[np.ndarray] = None,
    ) -> "Room":
        """Generate a room from width, depth, and height."""
        pt1 = np.array((0, 0, 0)) if origin is None else origin
        pt2 = pt1 + np.array((width, 0, 0))
        pt3 = pt2 + np.array((0, depth, 0))
        floor = Polygon.rectangle3pts(pt1, pt2, pt3)
        _, ceiling, swall, ewall, nwall, wwall = floor.extrude(
            np.array((0, 0, floor_floor))
        )
        ceiling = ceiling.move(np.array((0, 0, floor_ceiling - floor_floor)))
        return cls(
            Surface(floor),
            Surface(ceiling),
            Surface(swall),
            Surface(ewall),
            Surface(nwall),
            Surface(wwall),
        )

    @property
    def primitives(self):
        """."""
        return [
            *self.materials.values(),
            *self.floor.primitives,
            *self.ceiling.primitives,
            *self.swall.primitives,
            *self.ewall.primitives,
            *self.nwall.primitives,
            *self.wwall.primitives,
        ]

    @property
    def window_primitives(self):
        """."""
        return [
            *[prim for srf in self.ceiling.windows for prim in srf.primitives],
            *[prim for srf in self.swall.windows for prim in srf.primitives],
            *[prim for srf in self.ewall.windows for prim in srf.primitives],
            *[prim for srf in self.nwall.windows for prim in srf.primitives],
            *[prim for srf in self.wwall.windows for prim in srf.primitives],
        ]

    def get_material_names(self) -> List[str]:
        """Get material identifiers."""
        return [prim.identifier for prim in self.materials.values()]

    def add_material(self, primitive) -> None:
        """Add a material to the material library."""
        self.materials[primitive.identifier] = primitive

    def validate(self) -> None:
        """Validate the room model."""
        for prim in [
            *self.floor.primitives,
            *self.ceiling.primitives,
            *self.swall.primitives,
            *self.ewall.primitives,
            *self.nwall.primitives,
            *self.wwall.primitives,
        ]:
            if prim.modifier not in self.materials:
                raise ValueError(
                    f"Unknown modifier {prim.modifier} in {prim.identifier}"
                )

    def rotate(self, deg):
        """Rotate the room counter clock-wise."""
        self.floor.rotate(deg)
        self.ceiling.rotate(deg)
        self.swall.rotate(deg)
        self.ewall.rotate(deg)
        self.wwall.rotate(deg)
        self.nwall.rotate(deg)



def thicken(base, thickness) -> List[Polygon]:
    """Thicken the surface."""
    direction = base.normal * thickness
    polygons = base.extrude(direction)
    # Remove duplicates.
    counts = [polygons.count(plg) for plg in polygons]
    polygons = [plg for plg, cnt in zip(polygons, counts) if cnt == 1]
    return polygons


def make_window(
    base: Polygon,
    vertices: np.ndarray,
    vec1: np.ndarray,
    vec2: np.ndarray,
    dist_left: float,
    dist_bot: float,
    width: float,
    height: float
) -> Tuple[Polygon, WindowSurface]:
    """Make a window and punch a hole."""
    win_pt1 = vertices[0] + vec1 * dist_bot + vec2 * dist_left
    win_pt2 = win_pt1 + vec1 * height
    win_pt3 = win_pt1 + vec2 * width
    window_polygon = Polygon.rectangle3pts(win_pt3, win_pt1, win_pt2)
    new_base = base - window_polygon
    return new_base, WindowSurface(
        polygon=window_polygon,
        primitive=utils.polygon_primitive(
            polygon=window_polygon,
            modifier="glass_60",
            identifier="void",
        )
    )

def make_window_wwr(
    base: Polygon,
    wwr: float
) -> Tuple[Polygon, WindowSurface]:
    window_polygon = base.scale(np.array((wwr, wwr, wwr)), base.centroid)
    base = base - window_polygon
    return base, WindowSurface(
        polygon=window_polygon,
        primitive=utils.polygon_primitive(
            polygon=window_polygon,
            modifier="glass_60",
            identifier="void",
        )
    )


def create_surface(
    base: Polygon,
    thickness: float = 0,
    modifier: str = "void",
    identifier: str = "void",
    wpd: Optional[List[List[float]]] = None,
    wwr: Optional[float] = None,
) -> Surface2:
    vertices = base.vertices
    vec1 = (vertices[1] - vertices[0]) / np.linalg.norm(
        vertices[1] - vertices[0]
    )
    vec2 = (vertices[2] - vertices[1]) / np.linalg.norm(
        vertices[2] - vertices[1]
    )
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
        for pd in wpd:
            base, _window = make_window(base, vertices, vec1, vec2, *pd)
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
    return Surface2(
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
) -> Room2:
    materials = utils.material_lib()
    pt1 = np.array((0, 0, 0))
    pt2 = pt1 + np.array((width, 0, 0))
    pt3 = pt2 + np.array((0, depth, 0))
    base_floor = Polygon.rectangle3pts(pt1, pt2, pt3)
    _, base_ceiling, base_swall, base_ewall, base_nwall, base_wwall = base_floor.extrude(
        np.array((0, 0, floor_floor))
    )
    base_ceiling = base_ceiling.move(np.array((0, 0, floor_ceiling - floor_floor)))
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
    return Room2(
        floor,
        ceiling,
        swall,
        ewall,
        nwall,
        wwall,
        materials.values(),
    )


def make_room(
    width: float,
    depth: float,
    floor_floor: float,
    floor_ceiling: float,
    windows,
    swall_thickness=None,
):
    """Make a side-lit shoebox room as a Room object."""
    aroom = Room.from_wdh(width, depth, floor_floor, floor_ceiling)
    if windows is not None:
        for window in windows:
            aroom.swall.make_window(*window)
        for window in aroom.swall.windows:
            window.modifier = "glass_60"
    if swall_thickness is not None:
        aroom.swall.thicken(swall_thickness)
    aroom.swall.modifier = "neutral_lambertian_0.5"
    aroom.ewall.modifier = "neutral_lambertian_0.5"
    aroom.nwall.modifier = "neutral_lambertian_0.5"
    aroom.wwall.modifier = "neutral_lambertian_0.5"
    aroom.ceiling.modifier = "neutral_lambertian_0.7"
    aroom.floor.modifier = "neutral_lambertian_0.2"
    return aroom
