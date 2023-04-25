# frads/types.py
"""
This module contains all data types used across frads.
The exceptions are the Vector and Polygon class in the geom.py module.

"""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import pyradiance as pr
from frads.geom import Polygon
from frads.geom import Vector


class PaneProperty(NamedTuple):
    """Window pane property object.

    Attributes:
        name str: material name.
        thickness float: pane thickness.
        gtype str: material type.
        coated_side str: coated side, front or back.
        wavelength: Wavelength data.
        transmittance: Transmittance data.
        reflectance_front: Reflectance front data.
        reflectance_back: Reflectance back data.
    """

    name: str
    thickness: float
    gtype: str
    coated_side: str
    wavelength: List[float]
    transmittance: List[float]
    reflectance_front: List[float]
    reflectance_back: List[float]

    def get_tf_str(self) -> str:
        wavelength_tf = tuple(zip(self.wavelength, self.transmittance))
        return "\n".join([" ".join(map(str, pair)) for pair in wavelength_tf])

    def get_rf_str(self) -> str:
        wavelength_rf = tuple(zip(self.wavelength, self.reflectance_front))
        return "\n".join([" ".join(map(str, pair)) for pair in wavelength_rf])

    def get_rb_str(self) -> str:
        wavelength_rb = tuple(zip(self.wavelength, self.reflectance_back))
        return "\n".join([" ".join(map(str, pair)) for pair in wavelength_rb])


class PaneRGB(NamedTuple):
    """Pane color data object.

    Attributes:
        measured_data: measured data as a PaneProperty object.
        coated_rgb: Coated side RGB.
        glass_rgb: Non-coated side RGB.
        trans_rgb: Transmittance RGB.
    """

    measured_data: PaneProperty
    coated_rgb: Iterable[float]
    glass_rgb: Iterable[float]
    trans_rgb: Iterable[float]


class MradModel(NamedTuple):
    """Mrad model object.
    Attributes:
        name: Model name
        material_path: Material path
        window_groups: Window primitives grouped by files
        window_normals: Window normals
        sender_grid: Grid ray samples mapped to grid surface name.
        sender_view: View ray samples mapped to view name.
        views: Mapping from View name to view properties.
        receiver_sky: Sky as the receiver object.
        bsdf_xml: Mapping from window groupd name to BSDF file path.
        cfs_paths: The list of files used for direct-sun coefficient calculations.
        ncp_shades: The list of non-coplanar shading files.
        black_env_path: Blackened environment file path.

    """

    name: str
    material_path: Path
    window_groups: Dict[str, List[pr.Primitive]]
    window_normals: List[Vector]
    sender_grid: dict
    sender_view: dict
    views: dict
    bsdf_xml: dict
    cfs_paths: list
    ncp_shades: dict
    black_env_path: Path


@dataclass
class MradPath:
    """
    This dataclass object holds all the paths during a mrad run.
    All attributes are initiated with default_factory set to the attribute's type,
    which means this object can be instantiated without any arguments and
    add define its attributes later.

    Attributes:
        pvmx: Point view matrix paths mapped to grid name.
        pvmxd: Direct only point view matrix paths mapped to grid name.
        pdsmx: Point daylight coefficient matrix file paths mapped to grid name.
        pcdsmx: Point direct-sun coefficient matrix file paths mapped to grid name.
        vvmx: View view matrix paths mapped to view name.
        vvmxd: Direct only view view matrix paths mapped to view name.
        vdsmx: View daylight coefficient matrix file paths mapped to grid name.
        vcdsmx: View direct-sun coefficient matrix file paths mapped to view name.
        vcdfmx: View direct-sun coefficient(f) matrix file paths mapped to view name.
        vcdrmx: View direct-sun coefficient(r) matrix file paths mapped to view name.
        vmap: View matrix material map mapped to view name.
        cdmap: Direct-sun matrix material map mapped to view name.
        dmx: Daylight matrix file paths mapped to window name.
        dmxd: Direct daylight matrix mapped to window name.
        smxd: Sun-only (3-4 sun patches) sky matrix file path.
        smx: sky matrix file path.
        smx_sun: Sun-only (one sun patch) sky matrix file path for illuminance.
        smx_sun_img: Sun-only (one sun pathc) sky matrix file path for rendering.
    """

    pvmx: Dict[str, Path] = field(default_factory=dict)
    vvmx: Dict[str, Path] = field(default_factory=dict)
    dmx: Dict[str, Path] = field(default_factory=dict)
    pdsmx: Dict[str, Path] = field(default_factory=dict)
    vdsmx: Dict[str, Path] = field(default_factory=dict)
    pcdsmx: Dict[str, Path] = field(default_factory=dict)
    vcdsmx: Dict[str, Path] = field(default_factory=dict)
    vcdfmx: Dict[str, Path] = field(default_factory=dict)
    vcdrmx: Dict[str, Path] = field(default_factory=dict)
    vmap: Dict[str, Path] = field(default_factory=dict)
    cdmap: Dict[str, Path] = field(default_factory=dict)
    smxd: Path = field(default_factory=Path)
    pvmxd: Dict[str, Path] = field(default_factory=dict)
    vvmxd: Dict[str, Path] = field(default_factory=dict)
    dmxd: Dict[str, Path] = field(default_factory=dict)
    smx: Path = field(default_factory=Path)
    smx_sun: Path = field(default_factory=Path)
    smx_sun_img: Path = field(default_factory=Path)


@dataclass
class NcpModel:
    """Non-coplanar data model."""

    windows: Sequence[pr.Primitive]
    ports: Sequence[pr.Primitive]
    env: Iterable[Path]
    sbasis: str
    rbasis: str


@dataclass
class EPlusWindowGas:
    """EnergyPlus Window Gas material data container."""

    name: str
    thickness: float
    type: list
    percentage: list
    primitive: str = ""


@dataclass
class EPlusOpaqueMaterial:
    """EnergyPlus Opaque material data container."""

    name: str
    roughness: str
    solar_absorptance: float
    visible_absorptance: float
    visible_reflectance: float
    primitive: pr.Primitive
    thickness: float = 0.0


@dataclass
class EPlusWindowMaterial:
    """EnergyPlus regular window material data container."""

    name: str
    visible_transmittance: float
    primitive: pr.Primitive


@dataclass
class EPlusWindowMaterialComplexShade:
    """EnergyPlus complex window material data container."""

    name: str
    layer_type: str
    thickness: float
    conductivity: float
    ir_transmittance: float
    front_emissivity: float
    back_emissivity: float
    top_opening_multiplier: float
    bottom_opening_multiplier: float
    left_side_opening_multiplier: float
    right_side_opening_multiplier: float
    front_opening_multiplier: float


@dataclass
class EPlusConstruction:
    """EnergyPlus construction data container."""

    name: str
    type: str
    layers: list


@dataclass
class EPlusOpaqueSurface:
    """EnergyPlus opaque surface data container."""

    name: str
    type: str
    polygon: Polygon
    construction: str
    boundary: str
    sun_exposed: bool
    zone: str
    fenestrations: list


@dataclass
class EPlusFenestration:
    """EnergyPlus fenestration data container."""

    name: str
    type: str
    polygon: Polygon
    construction: EPlusConstruction
    host: EPlusOpaqueSurface


@dataclass
class EPlusZone:
    """EnergyPlus zone data container."""

    name: str
    wall: Dict[str, EPlusOpaqueSurface]
    ceiling: Dict[str, EPlusOpaqueSurface]
    roof: Dict[str, EPlusOpaqueSurface]
    floor: Dict[str, EPlusOpaqueSurface]
    window: Dict[str, EPlusFenestration]


class ViewType:
    perspective = "v"
    parallel = "l"
    cylindrical = "c"
    hemispherical_fisheye = "h"
    angular_fisheye = "a"
    planisphere_fishsye = "s"


@dataclass
class View:
    """View data object."""

    position: Vector
    direction: Vector
    up: Vector = Vector(0, 0, 1)
    vtype: str = "v"
    vert: float = 45
    hori: float = 45
    aft: float = 0
    fore: float = 0
    shift: float = 0
    lift: float = 0
    xres: int = 256
    yres: int = 256

    def __str__(self) -> str:
        return (
            f"-vt{self.vtype} "
            f"-vp {self.position} -vd {self.direction} "
            f"-vu {self.up} "
            f"-vv {self.vert} -vh {self.hori}"
            f"-x {self.xres} -y {self.yres}"
        )

    def args(self) -> list:
        return [
            f"-vt{self.vtype}",
            "-vp",
            *map(str, self.position.to_list()),
            "-vd",
            *map(str, self.direction.to_list()),
            "-vu",
            *map(str, self.up.to_list()),
            "-vv",
            str(self.vert),
            "-vh",
            str(self.hori),
            "-vo",
            str(self.fore),
            "-va",
            str(self.aft),
            "-vs",
            str(self.shift),
            "-vl",
            str(self.lift),
            "-x",
            str(self.xres),
            "-y",
            str(self.yres),
        ]


class ColorPrimaries(NamedTuple):
    xr: float
    yr: float
    xg: float
    yg: float
    xb: float
    yb: float
    xw: float
    yw: float


class IntArg:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value) -> None:
        if not isinstance(value, int):
            raise ValueError("ab has to be an integer")
        self.value = value
        obj.__dict__[self.name] = value


class FloatArg:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value) -> None:
        if not isinstance(value, float):
            raise ValueError("Value has to be a float")
        self.value = value
        obj.__dict__[self.name] = value


class TupleArg:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value) -> None:
        if not isinstance(value, tuple):
            raise ValueError("Value has to be a tuple")
        if not all(isinstance(i, (int, float)) for i in value):
            raise ValueError("Value inside has to be a integer or float")
        self.value = value
        obj.__dict__[self.name] = value


class Options:
    aa: float = FloatArg()
    ab: int = IntArg()
    ad: int = IntArg()
    ar: int = IntArg()
    as_: int = IntArg()
    av: Tuple[float, float, float] = TupleArg()
    aw: int = IntArg()
    dc: float = FloatArg()
    dj: float = FloatArg()
    dr: int = IntArg()
    dp: int = IntArg()
    ds: float = FloatArg()
    dt: float = FloatArg()
    lr: int = IntArg()
    lw: float = FloatArg()
    ms: float = FloatArg()
    pa: float = FloatArg()
    pj: float = FloatArg()
    ps: int = IntArg()
    pt: float = FloatArg()
    ss: float = IntArg()
    st: float = FloatArg()
    x: int = IntArg()
    y: int = IntArg()
    i: bool = False
    I: bool = False

    def args(self):
        arg_list: List[str] = []
        for k, v in self.__dict__.items():
            if k in Options.__annotations__:
                if isinstance(v, bool):
                    sign = "+" if v else "-"
                    arg_list.append(f"-{k}{sign}")
                elif isinstance(v, (int, float)):
                    arg_list.append("-" + k)
                    arg_list.append(str(v))
                elif isinstance(v, tuple):
                    arg_list.append("-" + k)
                    arg_list.extend(map(str, v))
        return arg_list
