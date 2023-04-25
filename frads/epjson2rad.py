"""Convert an EnergyPlus epJSON file into Radiance model[s]."""

from configparser import ConfigParser
import logging
import math
import json
import os
from pathlib import Path
import sys
import subprocess as sp
from typing import Dict, List

import pyradiance as pr

from frads import geom, utils
from frads.types import (
    EPlusWindowGas,
    EPlusOpaqueMaterial,
    EPlusWindowMaterial,
    EPlusWindowMaterialComplexShade,
    EPlusConstruction,
    EPlusOpaqueSurface,
    EPlusFenestration,
    EPlusZone,
)


logger: logging.Logger = logging.getLogger("frads.epjson2rad")


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
    direction = surface.normal.scale(thickness)
    facade = surface.extrude(direction)[:2]
    for window in windows:
        facade.extend(window.extrude(direction)[2:])
    uniq = facade.copy()
    for idx, val in enumerate(facade):
        for rep in facade[:idx] + facade[idx + 1 :]:
            if set(val.to_list()) == set(rep.to_list()):
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


def check_outward(polygon: geom.Polygon, zone_center: geom.Vector) -> bool:
    """Check whether a surface is facing outside."""
    outward = True
    angle2center = polygon.normal.angle_from(zone_center - polygon.centroid)
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
        surface_primitives[name]["window"] = []
        surface_primitives[name]["xml"] = []
        surface_polygon = surface.polygon
        if not check_outward(surface_polygon, zone_center):
            surface_polygon = surface_polygon.flip()
        window_polygons = []
        for fen in surface.fenestrations:
            if constructions[fen.construction].type == "cfs":
                window_material = "void"
                surface_primitives[name]["xml"].append(fen.construction + ".xml")
            else:
                # Get the last construction layer
                window_material = (
                    constructions[fen.construction].layers[0].replace(" ", "_")
                )
            window_polygon = fen.polygon
            if check_outward(fen.polygon, zone_center):
                window_polygon = window_polygon.flip()
            window_polygons.append(window_polygon)
            surface_primitives[name]["window"].append(
                utils.polygon2prim(window_polygon, window_material, fen.name)
            )
        surface_primitives[name]["surface"] = []
        surface_construction_layers = constructions[surface.construction].layers
        # Second to the last of construction layers
        inner_material = surface_construction_layers[
            -min(2, len(surface_construction_layers))
        ].replace(" ", "_")
        surface_primitives[name]["surface"].append(
            utils.polygon2prim(surface_polygon, inner_material, surface.name)
        )
        if surface.sun_exposed:
            outer_material = surface_construction_layers[-1].replace(" ", "_")
            thickness = get_construction_thickness(
                constructions[surface.construction], materials
            )
            facade = thicken(surface_polygon, window_polygons, thickness)
            surface_primitives[name]["surface"].append(
                utils.polygon2prim(facade[1], outer_material, f"ext_{surface.name}")
            )
            for idx in range(2, len(facade)):
                surface_primitives[name]["surface"].append(
                    utils.polygon2prim(
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
        bsdf = {key: [v["value"] for v in epjs["Matrix:TwoDimension"][name]['values']]
                for key, name in names.items()}
        ncs = {key: epjs["Matrix:TwoDimension"][name]["number_of_columns"]
                for key, name in names.items()}
        nrs = {key: epjs["Matrix:TwoDimension"][name]["number_of_rows"]
                for key, name in names.items()}
        matrices[key] = {key: {"ncolumns": ncs[key], "nrows": nrs[key], "values": bsdf[key]} 
                         for key in names}
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
            [geom.Vector(*vertice.values()) for vertice in surface["vertices"]]
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
                    geom.Vector(
                        fen[f"vertex_{i}_x_coordinate"],
                        fen[f"vertex_{i}_y_coordinate"],
                        fen[f"vertex_{i}_z_coordinate"],
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


def epjson2rad(epjs: dict, epw=None) -> None:
    """Command-line program to convert a energyplus model into a Radiance model."""
    # Setup file structure
    objdir = Path("Objects")
    objdir.mkdir(exist_ok=True)
    mtxdir = Path("Matrices")
    mtxdir.mkdir(exist_ok=True)

    site, zones, constructions, materials, matrices = parse_epjson(epjs)
    # building_name = epjs["Building"].popitem()[0].replace(" ", "_")
    building_name = list(epjs["Building"].keys())[0].replace(" ", "_")

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
                    _vis.__setattr__(_key[0]+_key[-1], _mtxpath)
                elif _key[1] == "s":
                    _sol.__setattr__(_key[0]+_key[-1], _mtxpath)
            basis = [i.name for i in pr.ABASELIST if i.nangles == val['tvf']['ncolumns']].pop()
            abr_basis = "".join(i[0].lower() for i in basis.decode().lstrip("LBNL/").split())
            with open(opath, "wb") as wtr:
                wtr.write(
                    pr.wrapbsdf(basis=abr_basis, inp=[_vis, _sol], unlink=True, n=key)
                )
            xml_paths[key] = str(opath)

    # Write material file
    material_path = os.path.join("Objects", f"materials{building_name}.mat")
    with open(material_path, "w", encoding="ascii") as wtr:
        for material in materials.values():
            wtr.write(str(material.primitive))

    # For each zone write primitves to files and create a config file
    for name, zone in zones.items():
        mrad_config = ConfigParser(allow_no_value=False)
        mrad_config["SimControl"] = {
            "vmx_basis": "kf",
            "vmx_opt": "-ab 5 -ad 65536 -lw 1e-5",
            "fmx_basis": "kf",
            "smx_basis": "r4",
            "dmx_opt": "-ab 2 -ad 128 -c 5000",
            "dsmx_opt": "-ab 7 -ad 16384 -lw 5e-5",
            "cdsmx_opt": "-ab 1",
            "cdsmx_basis": "r6",
            "ray_count": "1",
            "nprocess": "1",
            "separate_direct": "False",
            "overwrite": "False",
            "method": "",
        }
        mrad_config["Site"] = {
            "epw_path": epw,
            "latitude": site["latitude"],
            "longitude": site["longitude"],
            "daylight_hours_only": "True",
        }
        if epw is not None:
            mrad_config["Site"]["epw_path"] = epw
        primitives = epluszone2rad(zone, constructions, materials)
        scene = []
        windows = []
        window_xmls = []
        floors = []
        for primitive in primitives:
            write_primitives(primitive, "Objects")
            for _name, item in primitive.items():
                if item["surface"] != []:
                    scene.append(os.path.join("Objects", _name + ".rad"))
                if item["window"] != []:
                    windows.append(os.path.join("Objects", _name + "_window.rad"))
                if item["xml"] != []:
                    window_xmls.extend(
                        [os.path.join("Resources", xml) for xml in item["xml"]]
                    )
        # Get floors
        for primitive in primitives[-1]:
            floors.append(os.path.join("Objects", primitive + ".rad"))

        mrad_config["Model"] = {
            "material": material_path,
            "scene": "\n".join(scene),
            "windows": " ".join(windows),
            "window_xml": " ".join(window_xmls),
            "ncp_shade": "",
        }
        mrad_config["RaySender"] = {
            "grid_surface": " ".join(floors),
            "grid_spacing": "1",
            "grid_height": "0.75",
        }
        with open(f"{name}.cfg", "w", encoding="utf-8") as wtr:
            mrad_config.write(wtr)


def read_ep_input(fpath: Path) -> dict:
    """Load and parse input file into a JSON object.
    If the input file is in .idf fomart, use command-line
    energyplus program to convert it to epJSON format
    Args:
        fpath: input file path
    Returns:
        epjs: JSON object as a Python dictionary
    """
    epjson_path: Path
    if fpath.suffix == ".idf":
        cmd = ["energyplus", "--convert-only", str(fpath)]
        sp.run(cmd, check=True)
        epjson_path = Path(fpath.with_suffix(".epJSON").name)
        if not epjson_path.is_file():
            raise FileNotFoundError(f"Converted {str(epjson_path)} not found.")
    elif fpath.suffix == ".epJSON":
        epjson_path = fpath
    else:
        raise Exception(f"Unknown file type {fpath}")
    with open(epjson_path) as rdr:
        epjs = json.load(rdr)
    return epjs
