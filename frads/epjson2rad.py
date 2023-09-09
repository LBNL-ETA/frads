"""Convert an EnergyPlus epJSON file into Radiance model[s]."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import math

import pyradiance as pr
import numpy as np
import epmodel as epm
from frads import geom, utils
from frads.eprad import EnergyPlusModel


logger: logging.Logger = logging.getLogger("frads.epjson2rad")


# @dataclass
# class EPlusWindowGas:
#     """EnergyPlus Window Gas material data container."""
#
#     name: str
#     thickness: float
#     type: list
#     percentage: list
#     # primitive: str = ""


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


# @dataclass
# class EPlusWindowMaterialComplexShade:
#     """EnergyPlus complex window material data container."""
#
#     name: str
#     layer_type: str
#     thickness: float
#     conductivity: float
#     ir_transmittance: float
#     front_emissivity: float
#     back_emissivity: float
#     top_opening_multiplier: float
#     bottom_opening_multiplier: float
#     left_side_opening_multiplier: float
#     right_side_opening_multiplier: float
#     front_opening_multiplier: float
#     primitive: str = ""


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
    polygon: geom.Polygon
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
    polygon: geom.Polygon
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


def fenestration_to_polygon(fen: epm.FenestrationSurfaceDetailed) -> geom.Polygon:
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
    if fen.number_of_vertices == 4:
        vertices.append(
            np.array(
                (
                    fen.vertex_4_x_coordinate,
                    fen.vertex_4_y_coordinate,
                    fen.vertex_4_z_coordinate,
                )
            )
        )
    return geom.Polygon(vertices)


def surface_to_polygon(srf: epm.BuildingSurfaceDetailed) -> geom.Polygon:
    """Convert a building_surface_detailed surface to a polygon."""
    if srf.vertices is None:
        raise ValueError("Surface has no vertices.")
    vertices = [
        np.array((v.vertex_x_coordinate, v.vertex_y_coordinate, v.vertex_z_coordinate))
        for v in srf.vertices
    ]
    return geom.Polygon(vertices)


def thicken(
    surface: geom.Polygon, windows: List[geom.Polygon], thickness: float
) -> list:
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


def check_outward(polygon: geom.Polygon, zone_center: np.ndarray) -> bool:
    """Check whether a surface is facing outside."""
    outward = True
    angle2center = geom.angle_between(polygon.normal, zone_center - polygon.centroid)
    if angle2center < math.pi / 4:
        outward = False
    return outward


def eplus_surface2primitive(
    surfaces: dict, constructions, zone_center, materials
) -> dict:
    """Conert EPlusOpaqueSurface (and its windows) to Radiance primitives."""
    surface_primitives: dict = {}
    for _, surface in surfaces.items():
        name = surface.name
        surface_primitives[name] = {}
        surface_primitives[name]["window"] = {}
        surface_primitives[name]["xml"] = []
        surface_polygon = surface.polygon
        if not check_outward(surface_polygon, zone_center):
            surface_polygon = surface_polygon.flip()
        window_polygons = []
        for fen in surface.fenestrations:
            _fen = {}
            if constructions[fen.construction].type == "cfs":
                window_material = "void"
                _fen["cfs"] = fen.construction
            else:
                # Get the last construction layer
                window_material = (
                    constructions[fen.construction].layers[0].replace(" ", "_")
                )
            window_polygon = fen.polygon
            if check_outward(fen.polygon, zone_center):
                window_polygon = window_polygon.flip()
            window_polygons.append(window_polygon)
            _fen["data"] = utils.polygon_primitive(
                window_polygon, window_material, fen.name
            )
            surface_primitives[name]["window"][fen.name] = _fen
        surface_primitives[name]["surface"] = []
        surface_construction_layers = constructions[surface.construction].layers
        # Second to the last of construction layers
        inner_material = surface_construction_layers[
            -min(2, len(surface_construction_layers))
        ].replace(" ", "_")
        surface_primitives[name]["surface"].append(
            utils.polygon_primitive(surface_polygon, inner_material, surface.name)
        )
        if surface.sun_exposed:
            outer_material = surface_construction_layers[-1].replace(" ", "_")
            thickness = get_construction_thickness(
                constructions[surface.construction], materials
            )
            facade = thicken(surface_polygon, window_polygons, thickness)
            surface_primitives[name]["surface"].append(
                utils.polygon_primitive(
                    facade[1], outer_material, f"ext_{surface.name}"
                )
            )
            for idx in range(2, len(facade)):
                surface_primitives[name]["surface"].append(
                    utils.polygon_primitive(
                        facade[idx], inner_material, f"sill_{surface.name}.{idx}"
                    )
                )
    return surface_primitives


def write_primitives(surfaces: dict, odir: str) -> None:
    """Write surface and subsurface primitives."""
    for name, item in surfaces.items():
        opath = os.path.join(odir, name + ".rad")
        with open(opath, "w", encoding="ascii") as wtr:
            for primitive in item["surface"]:
                wtr.write(str(primitive))
        if item["window"] != []:
            opath = os.path.join(odir, name + "_window.rad")
            with open(opath, "w", encoding="ascii") as wtr:
                for primitive in item["window"]:
                    wtr.write(str(primitive))


def epluszone2rad(zone, constructions, materials):
    """Convert a EPlusZone object to a Radiance model."""
    zone_center = geom.polygon_center(*[wall.polygon for wall in zone.wall.values()])
    wall = eplus_surface2primitive(zone.wall, constructions, zone_center, materials)
    ceiling = eplus_surface2primitive(
        zone.ceiling, constructions, zone_center, materials
    )
    roof = eplus_surface2primitive(zone.roof, constructions, zone_center, materials)
    floor = eplus_surface2primitive(zone.floor, constructions, zone_center, materials)
    return wall, ceiling, roof, floor


def parse_material(name: str, material: epm.Material) -> EPlusOpaqueMaterial:
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


def parse_material_nomass(name: str, material: dict) -> EPlusOpaqueMaterial:
    """Parse EP Material:NoMass"""
    name = name.replace(" ", "_")
    roughness = material.get("roughness", "Smooth")
    solar_absorptance = material.get("solar_absorptance", 0.7)
    visible_absorptance = material.get("visible_absorptance", 0.7)
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


# def parse_windowmaterial_complexshade(
#     name: str, material: epm.WindowMaterialComplexShade
# ) -> EPlusWindowMaterialComplexShade:
#     """Parse EP WindowMaterial:ComplexShade."""
#     return EPlusWindowMaterialComplexShade(
#         name.replace(" ", "_"),
#         material.layer_type or "",
#         material.thickness or 0.002,
#         material.conductivity or 1.0,
#         material.ir_transmittance or 0.0,
#         material.front_emissivity or 0.84,
#         material.back_emissivity or 0.84,
#         material.top_opening_multiplier or 0.0,
#         material.bottom_opening_multiplier or 0.0,
#         material.left_side_opening_multiplier or 0.0,
#         material.right_side_opening_multiplier or 0.0,
#         material.front_opening_multiplier or 0.0,
#     )


# def parse_windowmaterial_gap(name: str, material: dict) -> EPlusWindowGas:
#     """Parse EP WindowMaterial:Gap"""
#     name = name.replace(" ", "_")
#     thickness = material["thickness"]
#     gas = material["gas_or_gas_mixture_"]
#     return EPlusWindowGas(name, thickness, gas, [1])
#
#
# def parse_windowmaterial_gas(name: str, material: dict) -> EPlusWindowGas:
#     """Parse EP WindowMaterial:Gas"""
#     name = name.replace(" ", "_")
#     ptype = [material["gas_type"]]
#     thickness = material["thickness"]
#     percentage = [1]
#     return EPlusWindowGas(name, thickness, ptype, percentage)


# def parse_windowmaterial_gasmixture(name: str, material: dict) -> EPlusWindowGas:
#     """Parse EP WindowMaterial:GasMixture"""
#     name = name.replace(" ", "_")
#     thickness = material["thickness"]
#     gas = [material["Gas"]]
#     percentage = [1]
#     return EPlusWindowGas(name, thickness, gas, percentage)


def parse_windowmaterial_simpleglazingsystem(
    name: str, material: dict
) -> EPlusWindowMaterial:
    """Parse EP WindowMaterial:SimpleGlazingSystem"""
    identifier = name.replace(" ", "_")
    shgc = material["solar_heat_gain_coefficient"]
    tmit = material.get("visible_transmittance", shgc)
    tmis = tmit2tmis(tmit)
    primitive = pr.Primitive("void", "glass", identifier, [], [tmis, tmis, tmis])
    return EPlusWindowMaterial(identifier, tmit, primitive)


def parse_windowmaterial_simpleglazing(
    name: str, material: dict
) -> EPlusWindowMaterial:
    """Parse EP WindowMaterial:Simpleglazing"""
    identifier = name.replace(" ", "_")
    tmit = material["visible_transmittance"]
    tmis = tmit2tmis(tmit)
    primitive = pr.Primitive("void", "glass", identifier, [], [tmis, tmis, tmis])
    return EPlusWindowMaterial(identifier, tmit, primitive)


def parse_windowmaterial_glazing(name: str, material: dict) -> EPlusWindowMaterial:
    """Parse EP WindowMaterial:Glazing"""
    identifier = name.replace(" ", "_")
    if material["optical_data_type"].lower() == "bsdf":
        tmit = 1
    else:
        tmit = material["visible_transmittance_at_normal_incidence"]
    tmis = tmit2tmis(tmit)
    primitive = pr.Primitive("void", "glass", identifier, [], [tmis, tmis, tmis])
    return EPlusWindowMaterial(identifier, tmit, primitive)


def parse_windowmaterial_blind(inp: dict) -> dict:
    """Parse EP WindowMaterial:Blind"""
    blind_prims = {}
    for key, val in inp.items():
        _id = key.replace(" ", "_")
        # back_beam_vis_refl = val['back_side_slat_beam_visible_reflectance']
        # back_diff_vis_refl = val['back_side_slat_diffuse_visible_reflectance']
        # front_beam_vis_refl = val['front_side_slat_beam_visible_reflectance']
        front_diff_vis_refl = val["front_side_slat_diffuse_visible_reflectance"]
        # slat_width = val['slat_width']
        # slat_thickness = val['slat_thickness']
        # slat_separation = val['slat_separation']
        # slat_angle = val['slat_angle']
        blind_prims[key] = pr.Primitive(
            "void",
            "plastic",
            _id,
            [],
            [front_diff_vis_refl, front_diff_vis_refl, front_diff_vis_refl, 0, 0],
        )
        # genblinds_cmd = f"genblinds {_id} {_id} {slat_width} 3
        # {20*slat_separation} {slat_angle}"
    return blind_prims


def parse_construction_complexfenestrationstate(epjs):
    """Parser EP Construction:ComplexFenestrationState."""
    construction = epjs.get("Construction:ComplexFenestrationState", {})
    cfs = {}
    matrices = {}
    for key, val in construction.items():
        val["ctype"] = "cfs"
        names = {
            "tvf": val["visible_optical_complex_front_transmittance_matrix_name"],
            "tvb": val["visible_optical_complex_back_transmittance_matrix_name"],
            # "rvf": val["visible_optical_complex_front_reflectance_matrix_name"],
            # "rvb": val["visible_optical_complex_back_reflectance_matrix_name"],
            "tsf": val["solar_optical_complex_front_transmittance_matrix_name"],
            # "tsb": val["solar_optical_complex_back_transmittance_matrix_name"],
            # "rsf": val["solar_optical_complex_front_reflectance_matrix_name"],
            "rsb": val["solar_optical_complex_back_reflectance_matrix_name"],
        }
        bsdf = {
            key: [v["value"] for v in epjs["Matrix:TwoDimension"][name]["values"]]
            for key, name in names.items()
        }
        ncs = {
            key: epjs["Matrix:TwoDimension"][name]["number_of_columns"]
            for key, name in names.items()
        }
        nrs = {
            key: epjs["Matrix:TwoDimension"][name]["number_of_rows"]
            for key, name in names.items()
        }
        matrices[key] = {
            key: {"ncolumns": ncs[key], "nrows": nrs[key], "values": bsdf[key]}
            for key in names
        }
        cfs[key] = EPlusConstruction(key, "cfs", [])
    return cfs, matrices


def parse_opaque_surface(surfaces: dict, fenestrations: dict) -> dict:
    """Parse opaque surface to a EPlusOpaqueSurface object."""
    opaque_surfaces = {}
    for name, surface in surfaces.items():
        identifier = name.replace(" ", "_")
        fenes = [fen for fen in fenestrations.values() if fen.host == name]
        ptype = surface["surface_type"]
        polygon = geom.Polygon(
            [np.array(list(vertice.values())) for vertice in surface["vertices"]]
        )
        for fen in fenes:
            polygon -= fen.polygon
        construction = surface["construction_name"]
        boundary = surface["outside_boundary_condition"]
        sun_exposed = surface["sun_exposure"] == "SunExposed"
        zone = surface["zone_name"]
        opaque_surfaces[name] = EPlusOpaqueSurface(
            identifier, ptype, polygon, construction, boundary, sun_exposed, zone, fenes
        )
    return opaque_surfaces


def parse_epjson_fenestration(
    fenes: Dict[str, epm.FenestrationSurfaceDetailed]
) -> Dict[str, EPlusFenestration]:
    """Parse fenestration dictionary to a EPlusFenestration object."""
    fenestrations = {}
    for name, fen in fenes.items():
        if fen.surface_type == epm.SurfaceType1.window:
            name = name.replace(" ", "_")
            host = fen.building_surface_name
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
            if fen.number_of_vertices == 4:
                vertices.append(
                    np.array(
                        (
                            fen.vertex_4_x_coordinate,
                            fen.vertex_4_y_coordinate,
                            fen.vertex_4_z_coordinate,
                        )
                    )
                )
            polygon = geom.Polygon(vertices)
            fenestrations[name] = EPlusFenestration(
                name, fen.surface_type, polygon, fen.construction_name, host
            )
    return fenestrations


def parse_epjson(epmodel: EnergyPlusModel) -> tuple:
    """
    Convert EnergyPlus JSON objects into Radiance primitives.
    """
    # parse each fenestration
    if epmodel.fenestration_surface_detailed is None:
        raise ValueError("No fenestration found in the model.")
    fenestrations = parse_epjson_fenestration(epmodel.fenestration_surface_detailed)

    # Get all the fenestration hosting surfaces
    fene_hosts = {val.host for val in fenestrations.values()}

    # parse each opaque surface
    opaque_surfaces = parse_opaque_surface(
        epjs["BuildingSurface:Detailed"], fenestrations
    )

    # parse each construction
    constructions = parse_construction(epjs["Construction"])

    if "Construction:WindowDataFile" in epjs:
        raise NotImplementedError("Construction:WindowDataFile is not supported yet.")
    if "Construction:WindowEquivalentLayer" in epjs:
        raise NotImplementedError(
            "Construction:WindowEquivalentLayer is not supported yet."
        )

    cfs, matrices = parse_construction_complexfenestrationstate(epjs)
    constructions.update(cfs)

    # parse materials
    materials = parse_epjson_material(epjs)

    # get exterior zones
    exterior_zones = [
        value.zone
        for key, value in opaque_surfaces.items()
        if (key in fene_hosts) and value.sun_exposed
    ]

    # get secondary zones, but we don't do anything with it yet.
    secondary_zones: dict = {}
    for key, value in opaque_surfaces.items():
        if (key in fene_hosts) and (value.zone not in exterior_zones):
            adjacent_zone = opaque_surfaces[value.boundary].zone
            if adjacent_zone in exterior_zones:
                secondary_zones[value.zone] = {}

    zones = {}
    # go through each exterior zone, update zone dictionary.
    for zname in exterior_zones:
        zone_name = zname.replace(" ", "_")
        surface_map: dict = {"Wall": {}, "Ceiling": {}, "Roof": {}, "Floor": {}}
        windows = {
            n: val
            for n, val in fenestrations.items()
            if opaque_surfaces[val.host].zone == zname
        }
        for name, surface in opaque_surfaces.items():
            if surface.zone == zname:
                surface_map[surface.type][name] = surface
        zones[zname] = EPlusZone(
            zone_name,
            surface_map["Wall"],
            surface_map["Ceiling"],
            surface_map["Roof"],
            surface_map["Floor"],
            windows,
        )
    return zones, constructions, materials, matrices


def epjson_to_rad(epmodel: EnergyPlusModel, epw=None) -> dict:
    """Command-line program to convert a energyplus model into a Radiance model.

    Args:
        epmodel (str): EnergyPlusModel file path.
        epw (str, optional): EnergyPlus weather file path. Defaults to None.

    Returns:
        A dictionary of Radiance model for each exterior zone.
    """
    site = get_dict_only_value(epmodel.site_location)
    zones, constructions, materials, matrices = parse_epjson(epmodel)

    if len(matrices) > 0:
        rsodir = Path("Resources")
        rsodir.mkdir(exist_ok=True)
        # Write matrix files to xml, if any
        xml_paths = {}
        for key, val in matrices.items():
            opath = rsodir / (key + ".xml")
            _vis = pr.WrapBSDFInput("Visible")
            _sol = pr.WrapBSDFInput("Solar")
            for _key, _val in val.items():
                _mtxpath = rsodir / f"{key}_{key}.mtx"
                with open(_mtxpath, "w") as fp:
                    fp.write(" ".join(str(v) for v in _val["values"]))
                if _key[1] == "v":
                    _vis.__setattr__(_key[0] + _key[-1], _mtxpath)
                elif _key[1] == "s":
                    _sol.__setattr__(_key[0] + _key[-1], _mtxpath)
            basis = [
                i.name for i in pr.ABASELIST if i.nangles == val["tvf"]["ncolumns"]
            ].pop()
            abr_basis = "".join(
                i[0].lower() for i in basis.decode().lstrip("LBNL/").split()
            )
            with open(opath, "wb") as wtr:
                wtr.write(
                    pr.wrapbsdf(basis=abr_basis, inp=[_vis, _sol], unlink=True, n=key)
                )
            xml_paths[key] = str(opath)

    rad_models = {}
    # For each zone write primitves to files and create a config file
    for name, zone in zones.items():
        radcfg = {}
        settings = {}
        model = {}
        # default to using three-phase method
        settings["method"] = "3"
        settings["sky_basis"] = "r1"
        if epw is not None:
            settings["epw_file"] = epw
        else:
            settings["latitude"] = epmodel.site_location["latitude"]
            settings["longitude"] = site["longitude"]
            settings["time_zone"] = ""
            settings["site_elevation"] = ""
        scene_data = []
        window_data = {}
        walls, ceilings, roofs, floors = epluszone2rad(zone, constructions, materials)
        for wall in walls.values():
            for srf in wall["surface"]:
                scene_data.append(srf.bytes)
            if wall["window"] != {}:
                for key, val in wall["window"].items():
                    window_data[key] = {"bytes": val["data"].bytes}
                    if "cfs" in val:
                        mtx = matrices[val["cfs"]]["tvb"]
                        nested = []
                        for i in range(0, len(mtx["values"]), mtx["nrows"]):
                            nested.append(mtx["values"][i : i + mtx["ncolumns"]])
                        window_data[key]["matrix_data"] = [
                            [[ele, ele, ele] for ele in row] for row in nested
                        ]
        for ceiling in ceilings.values():
            for srf in ceiling["surface"]:
                scene_data.append(srf.bytes)
            if ceiling["window"] != {}:
                for key, val in ceiling["window"].items():
                    window_data[key] = {"data": val["data"].bytes}
                    if "cfs" in val:
                        mtx = matrices[val["cfs"]]["tvb"]
                        nested = []
                        for i in range(0, len(mtx["values"]), mtx["nrows"]):
                            nested.append(mtx["values"][i : i + mtx["ncolumns"]])
                        window_data[key]["matrix_data"] = [nested, nested, nested]
        for roof in roofs.values():
            for srf in roof["surface"]:
                scene_data.append(srf.bytes)
            if roof["window"] != {}:
                for key, val in roof["window"].items():
                    window_data[key] = {"data": val["data"].bytes}
                    if "cfs" in val:
                        mtx = matrices[val["cfs"]]["tvb"]
                        nested = []
                        for i in range(0, len(mtx["values"]), mtx["nrows"]):
                            nested.append(mtx["values"][i : i + mtx["ncolumns"]])
                        window_data[key]["matrix_data"] = [nested, nested, nested]
        model["sensors"] = {}
        for floor in floors.values():
            for srf in floor["surface"]:
                scene_data.append(srf.bytes)
                _name = f"{name}_{srf.identifier}"
                polygon = utils.parse_polygon(srf)
                grid = utils.gen_grid(polygon, 0.76, 0.61)
                model["sensors"][_name] = {"data": grid}
        model["scene"] = {}
        model["views"] = {}
        model["scene"] = {"bytes": b" ".join(scene_data)}
        model["windows"] = window_data
        material_bytes = []
        for material in materials.values():
            if "primitive" in dir(material):
                material_bytes.append(material.primitive.bytes)
        model["materials"] = {"bytes": b" ".join(material_bytes)}
        radcfg["settings"] = settings
        radcfg["model"] = model
        rad_models[name] = radcfg

    return rad_models


class EnergyPlusModelParser:
    def __init__(self, ep_model: EnergyPlusModel):
        self.model = ep_model
        self._validate_ep_model()
        self.constructions = {}
        self.materials = {}

    def parse(self) -> Dict[str, dict]:
        self.constructions = self._parse_construction()
        self.materials = self._parse_materials()

        zones = {}
        for zname, _ in self.model.zone.items():
            zones[zname] = self._process_zone(zname)
        return zones

    def _parse_construction(self) -> dict:
        """Parse EnergyPlus construction"""
        if self.model.construction is None:
            raise ValueError("No construction found in the model.")
        constructions = {}
        for cname, clayer in self.model.construction.items():
            layers = list(clayer.model_dump(exclude_none=True).values())
            constructions[cname] = EPlusConstruction(cname, "default", layers)
        return constructions

    def _parse_materials(self) -> dict:
        """Parse EnergyPlus materials."""
        materials = {}
        material_key = [
            key for key in dir(self.model) if "material" in key.split("_")[0]
        ]
        for key in material_key:
            func = f"parse_{key}".lower()
            if func in globals() and (mdict := getattr(self.model, key)) is not None:
                print(key)
                tocall = globals()[func]
                for name, material in mdict.items():
                    materials[name.lower()] = tocall(name, material)
        return materials

    def _process_zone(self, zone_name: str) -> dict:
        """Process a zone given the zone name.

        Args:
            zone_name: Zone name.

        Returns:
            dict: A dictionary of scene (static surfaces) and windows.
        """
        scene: List[bytes] = []
        windows: Dict[str, bytes] = {}
        surfaces = self._collect_zone_surfaces(zone_name)
        fenestrations = self._collect_zone_fenestrations(surfaces)
        if (
            surfaces_fenestrations := self._pair_surfaces_fenestrations(
                surfaces, fenestrations
            )
        ) == {}:
            return {}
        surface_polygons = [surface_to_polygon(srf) for srf in surfaces.values()]
        center = geom.polygon_center(*surface_polygons)
        for sname, sdict in surfaces_fenestrations.items():
            _surface, _surface_fenestrations = self._process_surface(sname, sdict, center)
            scene.extend(_surface)
            windows = {**windows, **_surface_fenestrations}
        return {
            "scene": {"bytes": b" ".join(scene)},
            "windows": windows,
        }

    def _process_surface(
        self, surface_name: str, surface_fenestration: dict, zone_center: np.ndarray
    ) -> Tuple[List[bytes], Dict[str, bytes]]:
        """Process a surface in a zone.

        Args:
            surface_name: Surface name.
            surface_fenestration: A dictionary of surface and fenestration.
            zone_center: Zone center.

        Returns:
            Tuple[List[bytes], Dict[str, bytes]]: A tuple of scene and windows.
        """
        scene: List[bytes] = []
        windows: Dict[str, bytes] = {}
        opaque_surface = surface_fenestration["surface"]
        opaque_surface_polygon = surface_to_polygon(opaque_surface)
        if not check_outward(opaque_surface_polygon, zone_center):
            opaque_surface_polygon = opaque_surface_polygon.flip()
        for fname, fene in surface_fenestration["fenestration"].items():
            fenestration_polygon, window = self._process_fenestration(fname, fene)
            opaque_surface_polygon -= fenestration_polygon
        return scene, windows

    def _collect_zone_surfaces(self, zone_name: str) -> dict:
        if self.model.building_surface_detailed is None:
            return {}
        return {
            sname: srf
            for sname, srf in self.model.building_surface_detailed.items()
            if srf.zone_name == zone_name
        }

    def _collect_zone_fenestrations(self, zone_surfaces: dict) -> dict:
        if self.model.fenestration_surface_detailed is None:
            return {}
        return {
            fname: fen
            for fname, fen in self.model.fenestration_surface_detailed.items()
            if fen.building_surface_name in zone_surfaces
        }

    def _pair_surfaces_fenestrations(
        self, zone_surfaces: dict, zone_fenestrations: dict
    ):
        surface_fenestrations = {
            sname: {
                "surface": srf,
                "fenestration": {
                    fname: fen
                    for fname, fen in zone_fenestrations.items()
                    if fen.building_surface_name == sname
                },
            }
            for sname, srf in zone_surfaces.items()
        }
        return surface_fenestrations

    def _process_fenestration(
        self, name: str, fenestration: epm.FenestrationSurfaceDetailed
    ) -> Tuple[geom.Polygon, Dict[str, bytes]]:
        fenenstration_polygon = fenestration_to_polygon(fenestration)
        window_material = fenestration.construction_name
        window = {
            name: utils.polygon_primitive(
                fenenstration_polygon, window_material, name
            ).bytes
        }
        return fenenstration_polygon, window

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
    ep_model: EnergyPlusModel, epw_file: Optional[str] = None
) -> Optional[dict]:
    """Convert EnergyPlus model to Radiance model.

    Args:
        epmodel (EnergyPlusModel): EnergyPlus model.

    Returns:
        dict: Radiance model.
    """
    model_parser = EnergyPlusModelParser(ep_model)
    zone_models = model_parser.parse()
    settings = create_settings(ep_model, epw_file)
    rad_models = {
        zname: {"settings": settings, "model": zmodel}
        for zname, zmodel in zone_models.items()
    }

    return rad_models
