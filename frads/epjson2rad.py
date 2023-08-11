"""Convert an EnergyPlus epJSON file into Radiance model[s]."""

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import sys
from typing import Dict, List


import numpy as np
import pyradiance as pr
from frads import geom, utils


logger: logging.Logger = logging.getLogger("frads.epjson2rad")


@dataclass
class EPlusWindowGas:
    """EnergyPlus Window Gas material data container."""

    name: str
    thickness: float
    type: list
    percentage: list
    # primitive: str = ""


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
    primitive: str = ""


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


def tmit2tmis(tmit: float) -> float:
    """Convert from transmittance to transmissivity."""
    a = 0.0072522239
    b = 0.8402528435
    c = 0.9166530661
    d = 0.0036261119
    tmis = ((math.sqrt(a * tmit**2 + b) - c) / d) / tmit
    return max(0, min(tmis, 1))


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


def parse_material(name: str, material: dict) -> EPlusOpaqueMaterial:
    """Parser EP Material."""
    name = name.replace(" ", "_")
    roughness = material.get("roughness", "Smooth")
    thickness = material.get("thickness", 0.0)
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


def parse_windowmaterial_complexshade(
    name: str, material: dict
) -> EPlusWindowMaterialComplexShade:
    """Parse EP WindowMaterial:ComplexShade."""
    return EPlusWindowMaterialComplexShade(
        name.replace(" ", "_"),
        material.get("layer_type", ""),
        material.get("thickness", 0.002),
        material.get("conductivity", 1.0),
        material.get("ir_transmittance", 0.0),
        material.get("front_emissivity", 0.84),
        material.get("back_emissivity", 0.84),
        material.get("top_opening_multiplier", 0.0),
        material.get("bottom_opening_multiplier", 0.0),
        material.get("left_side_opening_multiplier", 0.0),
        material.get("right_side_opening_multiplier", 0.0),
        material.get("front_opening_multiplier", 0.0),
    )


def parse_windowmaterial_gap(name: str, material: dict) -> EPlusWindowGas:
    """Parse EP WindowMaterial:Gap"""
    name = name.replace(" ", "_")
    thickness = material["thickness"]
    gas = material["gas_or_gas_mixture_"]
    return EPlusWindowGas(name, thickness, gas, [1])


def parse_windowmaterial_gas(name: str, material: dict) -> EPlusWindowGas:
    """Parse EP WindowMaterial:Gas"""
    name = name.replace(" ", "_")
    ptype = [material["gas_type"]]
    thickness = material["thickness"]
    percentage = [1]
    return EPlusWindowGas(name, thickness, ptype, percentage)


def parse_windowmaterial_gasmixture(name: str, material: dict) -> EPlusWindowGas:
    """Parse EP WindowMaterial:GasMixture"""
    name = name.replace(" ", "_")
    thickness = material["thickness"]
    gas = [material["Gas"]]
    percentage = [1]
    return EPlusWindowGas(name, thickness, gas, percentage)


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


def parse_epjson_material(epjs: dict) -> dict:
    """Parse each material type."""
    materials = {}
    for key, value in epjs.items():
        if "material" in key.split(":")[0].lower():
            for name, material in value.items():
                tocall = globals()[f"parse_{key.replace(':', '_')}".lower()]
                materials[name.lower()] = tocall(name, material)
    return materials


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


def parse_construction(construction: dict) -> dict:
    """Parse EP Construction"""
    constructions = {}
    for cname, clayer in construction.items():
        layers = list(clayer.values())
        constructions[cname] = EPlusConstruction(cname, "default", layers)
    return constructions


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


def parse_epjson_fenestration(fenes: dict) -> Dict[str, EPlusFenestration]:
    """Parse fenestration dictionary to a EPlusFenestration object."""
    fenestrations = {}
    for name, fen in fenes.items():
        surface_type = fen["surface_type"]
        if surface_type != "Door":
            name = name.replace(" ", "_")
            host = fen["building_surface_name"]
            vertices = []
            for i in range(1, fen["number_of_vertices"] + 1):
                vertices.append(
                    np.array(
                        (
                            fen[f"vertex_{i}_x_coordinate"],
                            fen[f"vertex_{i}_y_coordinate"],
                            fen[f"vertex_{i}_z_coordinate"],
                        )
                    )
                )
            polygon = geom.Polygon(vertices)
            construction = fen["construction_name"]
            fenestrations[name] = EPlusFenestration(
                name, surface_type, polygon, construction, host
            )
    return fenestrations


def parse_epjson(epjs: dict) -> tuple:
    """
    Convert EnergyPlus JSON objects into Radiance primitives.
    """
    # get site information
    site = list(epjs["Site:Location"].values())[0]

    # parse each fenestration
    fenes = epjs.get("FenestrationSurface:Detailed")
    if fenes is not None:
        fenestrations = parse_epjson_fenestration(fenes)
    else:
        logger.warning("No fenestration found in the model.")
        sys.exit(1)

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
    return site, zones, constructions, materials, matrices


def epjson2rad(epmodel, epw=None) -> dict:
    """Command-line program to convert a energyplus model into a Radiance model."""
    site, zones, constructions, materials, matrices = parse_epjson(epmodel.epjs)
    # building_name = epjs["Building"].popitem()[0].replace(" ", "_")
    building_name = list(epmodel.epjs["Building"].keys())[0].replace(" ", "_")

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
            settings["latitude"] = site["latitude"]
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
