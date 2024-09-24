"""Convert an EnergyPlus epJSON file into Radiance model[s]."""

from dataclasses import dataclass
import logging
from typing import Dict, Optional, Any, List, Tuple
import math

import pyradiance as pr
import numpy as np
from frads.geom import (
    angle_between,
    Polygon,
    polygon_center,
)
from frads.utils import polygon_primitive, gen_grid
from frads.eplus_model import EnergyPlusModel
from epmodel.epmodel import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    Material,
    MaterialNoMass,
    SurfaceType,
    WindowMaterialGlazing,
    WindowMaterialSimpleGlazingSystem,
)


logger: logging.Logger = logging.getLogger("frads.epjson2rad")


OMEGAS = {
    "kf": [
        0.0238639257641843,
        0.02332285973188108,
        0.02332285973188108,
        0.02332285973188108,
        0.02332285973188108,
        0.02332285973188108,
        0.02332285973188108,
        0.02332285973188108,
        0.02332285973188108,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.021916319185673504,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.023622221568953158,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.022385166034409335,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.02238516603440936,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.019685184640794308,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.021916319185673487,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
        0.0175372336349361,
    ],
    "kh": [
        0.04025940809134382,
        0.038724862132931824,
        0.038724862132931824,
        0.038724862132931824,
        0.038724862132931824,
        0.038724862132931824,
        0.038724862132931824,
        0.038724862132931824,
        0.038724862132931824,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04640756737178026,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.04662852131277809,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.03866539339025036,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04533939830955456,
        0.04280163786238007,
        0.04280163786238007,
        0.04280163786238007,
        0.04280163786238007,
    ],
    "kq": [
        0.07688024442411853,
        0.07132814589069367,
        0.07132814589069367,
        0.07132814589069367,
        0.07132814589069367,
        0.07132814589069367,
        0.07132814589069367,
        0.07132814589069367,
        0.07132814589069367,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08150924303937566,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.08302065811560476,
        0.06496605352254502,
        0.06496605352254502,
        0.06496605352254502,
        0.06496605352254502,
        0.06496605352254502,
        0.06496605352254502,
        0.06496605352254502,
        0.06496605352254502,
    ],
}


@dataclass
class SurfaceWithNamedFenestrations:
    """A surface with fenestrations."""

    surface: BuildingSurfaceDetailed
    fenestrations: Dict[str, FenestrationSurfaceDetailed]


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
    thickness: float = 0.0


@dataclass
class EPlusConstruction:
    """EnergyPlus construction data container."""

    name: str
    type: str
    layers: list
    thickness: float


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


def get_dict_only_value(d: Optional[Dict]) -> Any:
    """Get the only value in a dictionary."""
    if d is None:
        raise ValueError("Object is None.")
    if len(d) != 1:
        raise ValueError("More than one value in the dictionary.")
    return next(iter(d.values()))


def tmit2tmis(tmit: float) -> float:
    """Convert from transmittance to transmissivity."""
    a = 0.0072522239
    b = 0.8402528435
    c = 0.9166530661
    d = 0.0036261119
    tmis = ((math.sqrt(a * tmit**2 + b) - c) / d) / tmit
    return max(0, min(tmis, 1))


def fenestration_to_polygon(fen: FenestrationSurfaceDetailed) -> Polygon:
    """Convert a fenestration_surface_detailed surface to a polygon."""
    vertices = [
        np.array(
            (
                fen.vertex_1_x_coordinate,
                fen.vertex_1_y_coordinate,
                fen.vertex_1_z_coordinate,
            )
        ),
        np.array(
            (
                fen.vertex_2_x_coordinate,
                fen.vertex_2_y_coordinate,
                fen.vertex_2_z_coordinate,
            )
        ),
        np.array(
            (
                fen.vertex_3_x_coordinate,
                fen.vertex_3_y_coordinate,
                fen.vertex_3_z_coordinate,
            )
        ),
    ]
    if fen.vertex_4_x_coordinate is not None:
        vertices.append(
            np.array(
                (
                    fen.vertex_4_x_coordinate,
                    fen.vertex_4_y_coordinate,
                    fen.vertex_4_z_coordinate,
                )
            )
        )
    return Polygon(vertices)


def surface_to_polygon(srf: BuildingSurfaceDetailed) -> Polygon:
    """Convert a building_surface_detailed surface to a polygon."""
    if srf.vertices is None:
        raise ValueError("Surface has no vertices.")
    vertices = [
        np.array(
            (
                v.vertex_x_coordinate,
                v.vertex_y_coordinate,
                v.vertex_z_coordinate,
            )
        )
        for v in srf.vertices
    ]
    return Polygon(vertices)


def thicken(
    surface: Polygon,
    windows: List[Polygon],
    thickness: float,
) -> List[Polygon]:
    """Thicken window-wall."""
    direction = surface.normal * thickness
    facade = surface.extrude(direction)[:2]
    for window in windows:
        facade.extend(window.extrude(direction)[2:])
    uniq = facade.copy()
    for idx, val in enumerate(facade):
        for rep in facade[:idx] + facade[idx + 1 :]:
            if set(map(tuple, val.vertices)) == set(map(tuple, rep.vertices)):
                uniq.remove(rep)
    return uniq


def get_construction_thickness(
    construction: EPlusConstruction, materials: dict
) -> float:
    """Get construction total thickness."""
    layer_thickness = []
    for layer in construction.layers:
        layer_thickness.append(materials[layer.lower()].thickness)
    return sum(layer_thickness)


def check_outward(polygon: Polygon, zone_center: np.ndarray) -> bool:
    """Check whether a surface is facing outside."""
    outward = True
    angle2center = angle_between(polygon.normal, zone_center - polygon.centroid)
    if angle2center < math.pi / 4:
        outward = False
    return outward


def parse_material(name: str, material: Material) -> EPlusOpaqueMaterial:
    """Parser EP Material."""
    name = name.replace(" ", "_")
    roughness = material.roughness.value
    thickness = material.thickness
    solar_absorptance = material.solar_absorptance or 0.7
    visible_absorptance = material.visible_absorptance or 0.7
    visible_reflectance = round(1 - visible_absorptance, 2)
    primitive = pr.Primitive(
        "void",
        "plastic",
        name,
        [],
        [visible_reflectance, visible_reflectance, visible_reflectance, 0, 0],
    )
    return EPlusOpaqueMaterial(
        name,
        roughness,
        solar_absorptance,
        visible_absorptance,
        visible_reflectance,
        primitive,
        thickness,
    )


def parse_material_no_mass(name: str, material: MaterialNoMass) -> EPlusOpaqueMaterial:
    """Parse EP Material:NoMass"""
    name = name.replace(" ", "_")
    roughness = material.roughness.value
    solar_absorptance = material.solar_absorptance or 0.7
    visible_absorptance = material.visible_absorptance or 0.7
    visible_reflectance = round(1 - visible_absorptance, 2)
    primitive = pr.Primitive(
        "void",
        "plastic",
        name,
        [],
        [visible_reflectance, visible_reflectance, visible_reflectance, 0, 0],
    )
    return EPlusOpaqueMaterial(
        name,
        roughness,
        solar_absorptance,
        visible_absorptance,
        visible_reflectance,
        primitive,
    )


def parse_window_material_simple_glazing_system(
    name: str, material: WindowMaterialSimpleGlazingSystem
) -> EPlusWindowMaterial:
    """Parse EP WindowMaterial:SimpleGlazingSystem"""
    identifier = name.replace(" ", "_")
    shgc = material.solar_heat_gain_coefficient
    tmit = material.visible_transmittance or shgc
    tmis = tmit2tmis(tmit)
    primitive = pr.Primitive("void", "glass", identifier, [], [tmis, tmis, tmis])
    return EPlusWindowMaterial(identifier, tmit, primitive)


def parse_window_material_glazing(
    name: str, material: WindowMaterialGlazing
) -> EPlusWindowMaterial:
    """Parse EP WindowMaterial:Glazing"""
    default_tmit = 0.6
    identifier = name.replace(" ", "_")
    if material.optical_data_type.value.lower() == "bsdf":
        tmit = 1
    else:
        tmit = material.visible_transmittance_at_normal_incidence or default_tmit
    tmis = tmit2tmis(tmit)
    primitive = pr.Primitive("void", "glass", identifier, [], [tmis, tmis, tmis])
    return EPlusWindowMaterial(identifier, tmit, primitive)


def parse_construction_complex_fenestration_state(
    epmodel: EnergyPlusModel,
) -> tuple:
    """Parser EP Construction:ComplexFenestrationState."""
    construction = epmodel.construction_complex_fenestration_state
    if construction is None:
        return {}, {}
    cfs = {}
    matrices = {}
    for key, val in construction.items():
        names = {
            "tvf": val.visible_optical_complex_front_transmittance_matrix_name,
            "rvb": val.visible_optical_complex_back_transmittance_matrix_name,
            "tsf": val.solar_optical_complex_front_transmittance_matrix_name,
            "rsb": val.solar_optical_complex_back_reflectance_matrix_name,
        }
        bsdf = {
            key: [v.value for v in epmodel.matrix_two_dimension[name].values]
            for key, name in names.items()
        }
        ncs = {
            key: epmodel.matrix_two_dimension[name].number_of_columns
            for key, name in names.items()
        }
        nrs = {
            key: epmodel.matrix_two_dimension[name].number_of_rows
            for key, name in names.items()
        }
        matrices[key] = {
            key: {"ncolumns": ncs[key], "nrows": nrs[key], "values": bsdf[key]}
            for key in names
        }
        cfs[key] = EPlusConstruction(key, "cfs", [], 0)
    return cfs, matrices


class EnergyPlusToRadianceModelConverter:
    """Convert EnergyPlus model to Radiance model."""

    def __init__(self, ep_model: EnergyPlusModel):
        self.model = ep_model
        self._validate_ep_model()
        self.constructions = {}
        self.materials = {}
        self.matrices = {}

    def parse(self) -> Dict[str, dict]:
        """Parse EnergyPlus model."""
        self.materials = self._parse_materials()
        self.constructions = self._parse_construction()

        zones = {zname: self._process_zone(zname) for zname in self.model.zone}

        return zones

    def _parse_construction(self) -> dict:
        """Parse EnergyPlus construction"""
        if self.model.construction is None:
            raise ValueError("No construction found in the model.")
        constructions = {}
        for cname, clayer in self.model.construction.items():
            layers = list(clayer.model_dump(exclude_none=True).values())
            constructions[cname] = EPlusConstruction(
                cname,
                "default",
                layers,
                sum(self.materials[layer.lower()].thickness for layer in layers),
            )
        cfs, matrices = parse_construction_complex_fenestration_state(self.model)
        for key, val in matrices.items():
            nested = []
            mtx = val["tvf"]
            for i in range(0, len(mtx["values"]), mtx["nrows"]):
                nested.append(mtx["values"][i : i + mtx["ncolumns"]])
            # Convert from BSDF to transmission matrix
            if mtx["ncolumns"] == 145:
                solid_angles = OMEGAS["kf"]
            elif mtx["ncolumns"] == 73:
                solid_angles = OMEGAS["kh"]
            elif mtx["ncolumns"] == 41:
                solid_angles = OMEGAS["kq"]
            else:
                raise KeyError("Unknown bsdf basis")
            assert len(solid_angles) == len(nested)
            assert len(solid_angles) == len(nested[0])
            self.matrices[key] = {
                "matrix_data": [
                    [
                        [ele * omg, ele * omg, ele * omg]
                        for ele, omg in zip(row, solid_angles)
                    ]
                    for row in nested
                ]
            }
        constructions.update(cfs)
        return constructions

    def _parse_materials(self) -> dict:
        """Parse EnergyPlus materials."""
        materials = {}
        material_keys = [
            "material",
            "material_no_mass",
            "window_material_simple_glazing_system",
            "window_material_glazing",
        ]
        for key in material_keys:
            func = f"parse_{key}".lower()
            if (mdict := getattr(self.model, key)) is not None:
                for name, material in mdict.items():
                    materials[name.lower()] = globals()[func](name, material)
        return materials

    def _process_zone(self, zone_name: str) -> dict:
        """Process a zone given the zone name.

        Args:
            zone_name: Zone name.

        Returns:
            dict: A dictionary of scene (static surfaces) and windows.
        """
        scene: List[bytes] = []
        windows: Dict[str, Dict[str, bytes]] = {}
        sensors: Dict[str, dict] = {}
        surfaces = self._collect_zone_surfaces(zone_name)
        fenestrations = self._collect_zone_fenestrations(surfaces)
        if fenestrations == {}:  # no fenestrations in the zone
            return {}
        surfaces_fenestrations = self._pair_surfaces_fenestrations(
            surfaces, fenestrations
        )
        surface_polygons = [surface_to_polygon(srf) for srf in surfaces.values()]
        center = polygon_center(*surface_polygons)
        view_direction = np.array([0.0, 0.0, 0.0])
        for sname, swnf in surfaces_fenestrations.items():
            opaque_surface_polygon = surface_to_polygon(swnf.surface)
            (
                _surface,
                _surface_fenestrations,
                window_polygons,
            ) = self._process_surface(
                sname,
                opaque_surface_polygon,
                swnf.surface.construction_name,
                swnf.fenestrations,
                center,
            )
            scene.extend(_surface)
            windows.update(_surface_fenestrations)
            if swnf.surface.surface_type == SurfaceType.floor:
                sensors[sname] = {
                    "data": gen_grid(
                        polygon=opaque_surface_polygon,
                        height=0.76,
                        spacing=0.61,
                    )
                }
            for window_polygon in window_polygons:
                view_direction += window_polygon.area
        view_direction *= -1
        view = pr.View(
            center.tolist(),
            view_direction.tolist(),
            "a",
            horiz=180,
            vert=180,
        )
        sensors[zone_name] = {"data": [center.tolist() + view_direction.tolist()]}

        return {
            "scene": {"bytes": b" ".join(scene)},
            "windows": windows,
            "materials": {
                "bytes": b" ".join(
                    mat.primitive.bytes for mat in self.materials.values()
                ),
                "matrices": self.matrices,
            },
            "sensors": sensors,
            "views": {zone_name: {"view": view}},
        }

    def _process_surface(
        self,
        surface_name: str,
        surface_polygon: Polygon,
        surface_construction_name: str,
        fenestrations: Dict[str, FenestrationSurfaceDetailed],
        zone_center: np.ndarray,
    ) -> Tuple[List[bytes], Dict[str, Dict[str, bytes]], List[Polygon]]:
        """Process a surface in a zone.

        Args:
            surface_name: Surface name.
            surface_polygon: Surface polygon.
            surface_construction_name: Surface construction name.
            fenestrations: Fenestrations in the surface.
            zone_center: Zone center.

        Returns:
            Tuple[List[bytes], Dict[str, Dict[str, bytes]], List[Polygon]]: A tuple of scene, windows, and window polygons.
        """
        scene: List[bytes] = []
        windows: Dict[str, dict] = {}
        window_polygons: List[Polygon] = []
        opaque_surface_name = surface_name.replace(" ", "_")
        if not check_outward(surface_polygon, zone_center):
            surface_polygon = surface_polygon.flip()
        for fname, fene in fenestrations.items():
            fenestration_polygon, window = self._process_fenestration(
                fname, fene, zone_center
            )
            window_polygons.append(fenestration_polygon)
            surface_polygon -= fenestration_polygon
            windows[fname] = {"bytes": window.bytes}
            if fene.construction_name in self.matrices:
                windows[fname]["matrix_name"] = fene.construction_name
        # polygon to primitive
        construction = self.constructions[surface_construction_name]
        inner_material_name = construction.layers[-1].replace(" ", "_")
        scene.append(
            polygon_primitive(
                surface_polygon, inner_material_name, opaque_surface_name
            ).bytes
        )

        # extrude the surface by thickness
        if fenestrations != {}:
            facade = thicken(surface_polygon, window_polygons, construction.thickness)
            outer_material_name = construction.layers[0].replace(" ", "_")
            scene.append(
                polygon_primitive(
                    facade[1], outer_material_name, f"ext_{opaque_surface_name}"
                ).bytes
            )
            for idx in range(2, len(facade)):
                scene.append(
                    polygon_primitive(
                        facade[idx],
                        inner_material_name,
                        f"sill_{opaque_surface_name}.{idx}",
                    ).bytes
                )

        return scene, windows, window_polygons

    def _collect_zone_surfaces(
        self, zone_name: str
    ) -> Dict[str, BuildingSurfaceDetailed]:
        if self.model.building_surface_detailed is None:
            return {}
        return {
            sname: srf
            for sname, srf in self.model.building_surface_detailed.items()
            if srf.zone_name == zone_name
        }

    def _collect_zone_fenestrations(
        self, zone_surfaces: dict
    ) -> Dict[str, FenestrationSurfaceDetailed]:
        if self.model.fenestration_surface_detailed is None:
            return {}
        return {
            fname: fen
            for fname, fen in self.model.fenestration_surface_detailed.items()
            if fen.building_surface_name in zone_surfaces
        }

    def _pair_surfaces_fenestrations(
        self,
        zone_surfaces: Dict[str, BuildingSurfaceDetailed],
        zone_fenestrations: Dict[str, FenestrationSurfaceDetailed],
    ) -> Dict[str, SurfaceWithNamedFenestrations]:
        surface_fenestrations = {}
        for sname, srf in zone_surfaces.items():
            named_fen = {}
            for fname, fen in zone_fenestrations.items():
                if fen.building_surface_name == sname:
                    named_fen[fname] = fen
            surface_fenestrations[sname] = SurfaceWithNamedFenestrations(srf, named_fen)
        return surface_fenestrations

    def _process_fenestration(
        self,
        name: str,
        fenestration: FenestrationSurfaceDetailed,
        zone_center: np.ndarray,
    ) -> Tuple[Polygon, pr.Primitive]:
        fenenstration_polygon = fenestration_to_polygon(fenestration)
        if check_outward(fenenstration_polygon, zone_center):
            fenenstration_polygon = fenenstration_polygon.flip()
        _construction = self.constructions[fenestration.construction_name]
        if _construction.type == "cfs":
            window_material = "void"
        else:
            window_material = _construction.layers[0].replace(" ", "_")
        fenestration_primitive = polygon_primitive(
            fenenstration_polygon, window_material, name.replace(" ", "_")
        )
        return fenenstration_polygon, fenestration_primitive

    def _validate_ep_model(self) -> bool:
        """Validate EnergyPlus model.
        Make sure the model has at least one zone, one building surface,
        and one fenestration.

        Returns:
            bool: True if the model is valid.
        """
        valid = True
        if self.model.zone is None:
            logger.warning("No zone found in the model.")
            valid = False
        if self.model.building_surface_detailed is None:
            logger.warning("No building surface found in the model.")
            valid = False
        if self.model.fenestration_surface_detailed is None:
            logger.warning("No fenestration found in the model.")
            valid = False
        return valid


def create_settings(ep_model: EnergyPlusModel, epw_file: Optional[str]) -> dict:
    """Create settings dictionary for Radiance model.

    Args:
        ep_model: EnergyPlus model.
        epw_file: EnergyPlus weather file path.

    Returns:
        dict: Radiance settings.
    """
    settings = {"method": "3", "sky_basis": "r1"}
    if epw_file:
        settings["epw_file"] = epw_file
    else:
        site = get_dict_only_value(ep_model.site_location)
        settings.update(
            {
                "latitude": site.latitude,
                "longitude": site.longitude,
                "time_zone": site.time_zone,
                "site_elevation": site.elevation,
            }
        )
    return settings


def epmodel_to_radmodel(
    ep_model: EnergyPlusModel,
    epw_file: Optional[str] = None,
    add_views: bool = True,
) -> dict:
    """Convert EnergyPlus model to Radiance models where each zone is a separate model.

    Args:
        ep_model: EnergyPlus model.
        epw_file: EnergyPlus weather file path. Defaults to None.
        add_views: Add views to the model. Such views will be positioned
            at the center of the zone facing windows weighted by window
            area. Defaults to True.

    Returns:
        dict: Radiance models.

    Examples:
        >>> radmodels = epmodel_to_radmodel(ep_model, epw_file)
    """
    model_parser = EnergyPlusToRadianceModelConverter(ep_model)
    zone_models = model_parser.parse()
    settings = create_settings(ep_model, epw_file)
    rad_models = {
        zname: {"settings": settings, "model": zmodel}
        for zname, zmodel in zone_models.items()
        if zmodel != {}
    }
    if not add_views:
        for model in rad_models.values():
            model["views"] = {}
    for model in rad_models.values():
        model["surfaces"] = {}

    return rad_models
