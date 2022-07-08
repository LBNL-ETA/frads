"""
This module contains all data types used across frads.
The exceptions are the Vector and Polygon class in the geom.py module.

"""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

from frads.geom import Polygon
from frads.geom import Vector

Matrix = List[List[float]]


class Primitive(NamedTuple):
    """
    Radiance Primitive, with attributes one-to-one mapped from Radiance.
    """

    modifier: str
    ptype: str
    identifier: str
    str_arg: str
    real_arg: str
    int_arg: str = "0"

    def __repr__(self) -> str:
        output = f"{self.modifier} {self.ptype} {self.identifier} "
        output += f"{self.str_arg} {self.int_arg} {self.real_arg} "
        return output

    def __str__(self) -> str:
        if "" in (self.modifier, self.ptype, self.identifier):
            return ""
        output = f"\n{self.modifier} {self.ptype} {self.identifier}\n"
        output += f"{self.str_arg}\n{self.int_arg}\n{self.real_arg}\n"
        return output


@dataclass(frozen=True)
class Sender:
    """
    Sender object for matrix generation.

    Attributes:
        form(str): types of sender, {surface(s)|view(v)|points(p)}
        sender(str): the sender object
        xres(int): sender x dimension
        yres(int): sender y dimension
    """

    form: str
    sender: Union[str, bytes]
    xres: Optional[int]
    yres: Optional[int]


@dataclass
class Receiver:
    """
    Receiver object for matrix generation.

    Attributes:
        receiver(str): receiver string which can be appended to one another
        basis(str): receiver basis, usually kf, r4, r6;
        modifier(str): modifiers to the receiver objects;
    """

    receiver: str
    basis: str
    modifier: str = ""

    def __add__(self, other):
        return Receiver(
            self.receiver + "\n" + other.receiver, self.basis, self.modifier
        )


@dataclass
class ScatteringData:
    """
    Scattering data object.

    Attributes:
        sdata(list[list[float]]): scattering data in nested lists.
        ncolumn(int): number of columns
        nrow(int): number of rows
    """

    sdata: Matrix
    ncolumn: int = field(init=False)
    nrow: int = field(init=False)

    def __post_init__(self):
        self.ncolumn = len(self.sdata[0])
        self.nrow = len(self.sdata)

    def __repr__(self) -> str:
        out = ""
        for row in self.sdata:
            for val in row:
                string = "%07.5f" % val
                out += string + "\t"
            out += "\n"
        return out

    def __str__(self) -> str:
        out = "#?RADIANCE\nNCOMP=3\n"
        out += "NROWS=%d\nNCOLS=%d\n" % (self.nrow, self.ncolumn)
        out += "FORMAT=ascii\n\n"
        for row in self.sdata:
            for val in row:
                string = "\t".join(["%07.5f" % val] * 3)
                out += string + "\t"
            out += "\n"
        return out


@dataclass
class BSDFData:
    """
    BSDF data object.

    Attributes:
        bsdf(list[list[float]]): BSDF data in nested lists.
        ncolumn(int): number of columns
        nrow(int): number of rows
    """

    bsdf: Matrix
    ncolumn: int = field(init=False)
    nrow: int = field(init=False)

    def __post_init__(self):
        self.ncolumn = len(self.bsdf[0])
        self.nrow = len(self.bsdf)


@dataclass(frozen=True)
class RadMatrix:
    """
    Radiance matrix object.

    Attributes:
        tf(ScatteringData): front-side transmission
        tb(ScatteringData): back-side transmission
    """

    tf: ScatteringData
    tb: ScatteringData


class PaneProperty(NamedTuple):
    """Window pane property object."""

    name: str
    thickness: float
    gtype: str
    coated_side: str
    wavelength: List[float]
    transmittance: List[float]
    reflectance_front: List[float]
    reflectance_back: List[float]

    def get_tf_str(self):
        wavelength_tf = tuple(zip(self.wavelength, self.transmittance))
        return "\n".join([" ".join(map(str, pair)) for pair in wavelength_tf])

    def get_rf_str(self):
        wavelength_rf = tuple(zip(self.wavelength, self.reflectance_front))
        return "\n".join([" ".join(map(str, pair)) for pair in wavelength_rf])

    def get_rb_str(self):
        wavelength_rb = tuple(zip(self.wavelength, self.reflectance_back))
        return "\n".join([" ".join(map(str, pair)) for pair in wavelength_rb])


class PaneRGB(NamedTuple):
    """Pane color data object."""

    measured_data: PaneProperty
    coated_rgb: List[float]
    glass_rgb: List[float]
    trans_rgb: List[float]


class WeaMetaData(NamedTuple):
    """Weather related meta data object."""

    city: str
    country: str
    latitude: float
    longitude: float
    timezone: int
    elevation: float

    def wea_header(self) -> str:
        header = f"place {self.city}_{self.country}\n"
        header += f"latitude {self.latitude}\n"
        header += f"longitude {self.longitude}\n"
        header += f"time_zone {self.timezone}\n"
        header += f"site_elevation {self.elevation}\n"
        header += "weather_data_file_units 1\n"
        return header


class WeaDataRow(NamedTuple):
    """Weather related data object."""

    month: int
    day: int
    hour: int
    minute: int
    second: int
    hours: float
    dni: float
    dhi: float

    def __str__(self):
        return f"{self.month} {self.day} {self.hours} {self.dni} {self.dhi}"

    def dt_string(self):
        return f"{self.month:02d}{self.day:02d}_{self.hour:02d}30"


class MradModel(NamedTuple):
    """Mrad model object."""

    material_path: str
    window_groups: Dict[str, List[Primitive]]
    window_normals: List[Vector]
    sender_grid: dict
    sender_view: dict
    views: dict
    receiver_sky: Receiver
    bsdf_xml: dict
    cfs_paths: list
    black_env_path: str


@dataclass
class MradPath:
    """
    This object holds all the paths during a mrad run.

    Attributes:
        smx(Path): sky matrix file path
    """

    smx: Optional[Path] = None
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
    smxd: Optional[Path] = None
    pvmxd: Dict[str, Path] = field(default_factory=dict)
    vvmxd: Dict[str, Path] = field(default_factory=dict)
    dmxd: Dict[str, Path] = field(default_factory=dict)
    smx_sun: Optional[Path] = None
    smx_sun_img: Optional[Path] = None


@dataclass
class EPlusWindowGas:
    name: str
    thickness: float
    type: list
    percentage: list
    primitive: str = ""


@dataclass
class EPlusOpaqueMaterial:
    name: str
    roughness: str
    solar_absorptance: float
    visible_absorptance: float
    visible_reflectance: float
    primitive: Primitive
    thickness: float = 0.0


@dataclass
class EPlusWindowMaterial:
    name: str
    visible_transmittance: float
    primitive: Primitive


@dataclass
class EPlusConstruction:
    name: str
    type: str
    layers: list


@dataclass
class EPlusOpaqueSurface:
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
    name: str
    type: str
    polygon: Polygon
    construction: EPlusConstruction
    host: EPlusOpaqueSurface


@dataclass
class EPlusZone:
    name: str
    wall: Dict[str, EPlusOpaqueSurface]
    ceiling: Dict[str, EPlusOpaqueSurface]
    roof: Dict[str, EPlusOpaqueSurface]
    floor: Dict[str, EPlusOpaqueSurface]
    window: Dict[str, EPlusFenestration]
