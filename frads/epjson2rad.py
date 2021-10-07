"""
Convert an EnergyPlus model as a parsed dictionary
to Radiance primitives.
TODO:
    * Parse window data file for Constradutilction:WindowDataFile

"""
import argparse
from configparser import ConfigParser
from dataclasses import dataclass
import json
import logging
import os
import subprocess as sp
from typing import Dict, List

from frads import radgeom, radutil, util

logger = logging.getLogger("frads.epjson2rad")


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
    primitive: radutil.Primitive
    thickness: float = 0.0


@dataclass
class EPlusWindowMaterial:
    name: str
    visible_transmittance: float
    primitive: radutil.Primitive


@dataclass
class EPlusConstruction:
    name: str
    type: str
    layers: list


@dataclass
class EPlusOpaqueSurface:
    name: str
    type: str
    polygon: radgeom.Polygon
    construction: str
    boundary: str
    sun_exposed: bool
    zone: str
    fenestrations: list


@dataclass
class EPlusFenestration:
    name: str
    type: str
    polygon: radgeom.Polygon
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


def thicken(surface: radgeom.Polygon,
            windows: List[radgeom.Polygon],
            thickness: float) -> list:
    """Thicken window-wall."""
    direction = surface.normal().scale(thickness)
    facade = surface.extrude(direction)[:2]
    [facade.extend(window.extrude(direction)[2:]) for window in windows]
    uniq = facade.copy()
    for idx, val in enumerate(facade):
        for rep in facade[:idx] + facade[idx + 1:]:
            if set(val.to_list()) == set(rep.to_list()):
                uniq.remove(rep)
    return uniq


def get_construction_thickness(
        construction: EPlusConstruction, materials: dict) -> float:
    """Get construction total thickness."""
    layer_thickness = []
    for layer in construction.layers:
        layer_thickness.append(materials[layer.lower()].thickness)
    return sum(layer_thickness)


def check_outward(polygon: radgeom.Polygon, zone_center: radgeom.Vector) -> bool:
    """Check whether a surface is facing outside."""
    pi = 3.14159265358579
    outward = True
    angle2center = polygon.normal().angle_from(
        zone_center - polygon.centroid())
    if angle2center < pi / 4:
        outward = False
    return outward


def eplus_surface2primitive(
        surfaces: dict, constructions, zone_center, materials) -> dict:
    """Conert EPlusOpaqueSurface (and its windows) to Radiance primitives."""
    surface_primitives = {}
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
            if constructions[fen.construction].type == 'cfs':
                window_material = "void"
                surface_primitives[name]["xml"].append(fen.construction + ".xml")
            else:
                # Get the last construction layer
                window_material = constructions[
                    fen.construction].layers[0].replace(" ", "_")
            window_polygon = fen.polygon
            if check_outward(fen.polygon, zone_center):
                window_polygon = window_polygon.flip()
            window_polygons.append(window_polygon)
            surface_primitives[name]["window"].append(
                radutil.polygon2prim(
                    window_polygon, window_material, fen.name))
        surface_primitives[name]["surface"] = []
        surface_construction_layers = constructions[surface.construction].layers
        # Second to the last of construction layers
        inner_material = surface_construction_layers[
            -min(2, len(surface_construction_layers))].replace(" ", "_")
        surface_primitives[name]["surface"].append(
            radutil.polygon2prim(surface_polygon, inner_material, surface.name))
        if surface.sun_exposed:
            outer_material = surface_construction_layers[-1].replace(" ", "_")
            thickness = get_construction_thickness(
                constructions[surface.construction], materials)
            facade = thicken(surface_polygon, window_polygons, thickness)
            surface_primitives[name]["surface"].append(
                radutil.polygon2prim(
                    facade[1], outer_material, f"ext_{surface.name}"))
            for idx in range(2, len(facade)):
                surface_primitives[name]["surface"].append(
                    radutil.polygon2prim(
                        facade[idx], inner_material, f"sill_{surface.name}.{idx}"))
    return surface_primitives


def write_primitives(surfaces: dict, odir: str):
    """Write surface and subsurface primitives."""
    for name, item in surfaces.items():
        opath = os.path.join(odir, name + ".rad")
        with open(opath, "w") as wtr:
            for primitive in item["surface"]:
                wtr.write(str(primitive))
        if item["window"] != []:
            opath = os.path.join(odir, name + "_window.rad")
            with open(opath, "w") as wtr:
                for primitive in item["window"]:
                    wtr.write(str(primitive))


def epluszone2rad(zone, constructions, materials):
    """Convert a EPlusZone object to a Radiance model."""
    zone_center = radgeom.polygon_center(
        *[wall.polygon for wall in zone.wall.values()])
    wall = eplus_surface2primitive(
        zone.wall, constructions, zone_center, materials)
    ceiling = eplus_surface2primitive(
        zone.ceiling, constructions, zone_center, materials)
    roof = eplus_surface2primitive(
        zone.roof, constructions, zone_center, materials)
    floor = eplus_surface2primitive(
        zone.floor, constructions, zone_center, materials)
    return wall, ceiling, roof, floor


def parse_material(name: str, material: dict) -> EPlusOpaqueMaterial:
    """Parser EP Material."""
    name = name.replace(" ", "_")
    roughness = material.get("roughness", "Smooth")
    thickness = material.get("thickness", 0.0)
    solar_absorptance = material.get("solar_absorptance", 0.7)
    visible_absorptance = material.get("visible_absorptance", 0.7)
    visible_reflectance = round(1 - visible_absorptance, 2)
    primitive = radutil.Primitive(
        "void", "plastic", name, "0",
        "5 {0} {0} {0} 0 0".format(visible_reflectance))
    return EPlusOpaqueMaterial(
        name, roughness, solar_absorptance, visible_absorptance,
        visible_reflectance, primitive, thickness)


def parse_material_nomass(name: str, material: dict) -> EPlusOpaqueMaterial:
    """Parse EP Material:NoMass"""
    name = name.replace(" ", "_")
    roughness = material.get("roughness", "Smooth")
    solar_absorptance = material.get("solar_absorptance", 0.7)
    visible_absorptance = material.get("visible_absorptance", 0.7)
    visible_reflectance = round(1 - visible_absorptance, 2)
    primitive = radutil.Primitive(
        "void", "plastic", name, "0",
        "5 {0} {0} {0} 0 0".format(visible_reflectance))
    return EPlusOpaqueMaterial(
        name, roughness, solar_absorptance,
        visible_absorptance, visible_reflectance, primitive)


def parse_windowmaterial_gap(name: str, material: dict) -> EPlusWindowGas:
    """Parse EP WindowMaterial:Gap"""
    name = name.replace(" ", "_")
    thickness = material["thickness"]
    gas = material["gas_or_gas_mixture_"]
    return EPlusWindowGas(name, thickness, gas, [1])


def parse_windowmaterial_gas(name: str, material: dict) -> EPlusWindowGas:
    """Parse EP WindowMaterial:Gas"""
    name = name.replace(" ", "_")
    type = [material["gas_type"]]
    thickness = material["thickness"]
    percentage = [1]
    return EPlusWindowGas(name, thickness, type, percentage)


def parse_windowmaterial_gasmixture(
        name: str, material: dict) -> EPlusWindowGas:
    """Parse EP WindowMaterial:GasMixture"""
    name = name.replace(" ", "_")
    thickness = material["thickness"]
    gas = [material["Gas"]]
    percentage = [1]
    return EPlusWindowGas(name, thickness, gas, percentage)


def parse_windowmaterial_simpleglazingsystem(
        name: str, material: dict) -> EPlusWindowMaterial:
    """Parse EP WindowMaterial:SimpleGlazingSystem"""
    identifier = name.replace(" ", "_")
    shgc = material["solar_heat_gain_coefficient"]
    tmit = material.get("visible_transmittance", shgc)
    tmis = util.tmit2tmis(tmit)
    primitive = radutil.Primitive(
        "void", "glass", identifier, "0",
        "3 {0:.2f} {0:.2f} {0:.2f}".format(tmis))
    return EPlusWindowMaterial(identifier, tmit, primitive)


def parse_windowmaterial_simpleglazing(
        name: str, material: dict) -> EPlusWindowMaterial:
    """Parse EP WindowMaterial:Simpleglazing"""
    identifier = name.replace(" ", "_")
    tmit = material['visible_transmittance']
    tmis = util.tmit2tmis(tmit)
    primitive = radutil.Primitive(
        "void", "glass", identifier, "0",
        "3 {0:.2f} {0:.2f} {0:.2f}".format(tmis))
    return EPlusWindowMaterial(identifier, tmit, primitive)


def parse_windowmaterial_glazing(
        name: str, material: dict) -> EPlusWindowMaterial:
    """Parse EP WindowMaterial:Glazing"""
    identifier = name.replace(" ", "_")
    if material['optical_data_type'].lower() == 'bsdf':
        tmit = 1
    else:
        tmit = material['visible_transmittance_at_normal_incidence']
    tmis = util.tmit2tmis(tmit)
    primitive = radutil.Primitive('void', 'glass', identifier, '0',
                                  "3 {0:.2f} {0:.2f} {0:.2f}".format(tmis))
    return EPlusWindowMaterial(identifier, tmit, primitive)


def parse_windowmaterial_blind(inp: dict) -> dict:
    """Parse EP WindowMaterial:Blind"""
    blind_prims = {}
    for key, val in inp.items():
        _id = key.replace(' ', '_')
        # back_beam_vis_refl = val['back_side_slat_beam_visible_reflectance']
        # back_diff_vis_refl = val['back_side_slat_diffuse_visible_reflectance']
        # front_beam_vis_refl = val['front_side_slat_beam_visible_reflectance']
        front_diff_vis_refl = val['front_side_slat_diffuse_visible_reflectance']
        # slat_width = val['slat_width']
        # slat_thickness = val['slat_thickness']
        # slat_separation = val['slat_separation']
        # slat_angle = val['slat_angle']
        blind_prims[key] = radutil.Primitive(
            'void', 'plastic', _id, '0',
            '5 {0:.2f} {0:.2f} {0:.2f} 0 0'.format(front_diff_vis_refl))
        # genblinds_cmd = f"genblinds {_id} {_id} {slat_width} 3 {20*slat_separation} {slat_angle}"
    return blind_prims


def parse_epjson_material(epjs: dict) -> dict:
    """Parse each material type."""
    materials = {}
    for key, value in epjs.items():
        if 'material' in key.split(':')[0].lower():
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
        val['ctype'] = "cfs"
        tf_name = val['visible_optical_complex_front_transmittance_matrix_name']
        tb_name = val['visible_optical_complex_back_transmittance_matrix_name']
        tf_list = epjs["Matrix:TwoDimension"][tf_name]['values']
        tb_list = epjs["Matrix:TwoDimension"][tb_name]['values']
        ncolumn = epjs["Matrix:TwoDimension"][tf_name]['number_of_columns']
        # if ncolumn < 145:
            # raise ValueError("BSDF resolution too low to take advantage of Radiance")
        tf_bsdf = util.nest_list([v['value'] for v in tf_list], ncolumn)
        tb_bsdf = util.nest_list([v['value'] for v in tb_list], ncolumn)
        tf = radutil.BSDFData(tf_bsdf).to_sdata()
        tb = radutil.BSDFData(tb_bsdf).to_sdata()
        matrices[key] = radutil.RadMatrix(tf, tb)
        cfs[key] = EPlusConstruction(key, "cfs", [])
    return cfs, matrices


def parse_construction(construction: dict) -> dict:
    """Parse EP Construction"""
    constructions = {}
    for cname, clayer in construction.items():
        layers = [layer for layer in clayer.values()]
        constructions[cname] = EPlusConstruction(cname, "default", layers)
    return constructions


def parse_opaque_surface(surfaces: dict, fenestrations: dict) -> dict:
    """Parse opaque surface to a EPlusOpaqueSurface object."""
    opaque_surfaces = {}
    for name, surface in surfaces.items():
        identifier = name.replace(" ", "_")
        fenes = [fen for fen in fenestrations.values() if fen.host == name]
        type = surface['surface_type']
        polygon = radgeom.Polygon(
            [radgeom.Vector(*vertice.values()) for vertice in surface['vertices']])
        for fen in fenes:
            polygon -= fen.polygon
        construction = surface['construction_name']
        boundary = surface['outside_boundary_condition']
        sun_exposed = surface['sun_exposure'] == 'SunExposed'
        zone = surface['zone_name']
        opaque_surfaces[name] = EPlusOpaqueSurface(
            identifier, type, polygon, construction,
            boundary, sun_exposed, zone, fenes)
    return opaque_surfaces


def parse_epjson_fenestration(fenes: dict) -> Dict[str, EPlusFenestration]:
    """Parse fenestration dictionary to a EPlusFenestration object."""
    fenestrations = {}
    for name, fen in fenes.items():
        surface_type = fen['surface_type']
        if surface_type != "Door":
            name = name.replace(" ", "_")
            host = fen['building_surface_name']
            vertices = []
            for i in range(1, fen['number_of_vertices'] + 1):
                vertices.append(
                    radgeom.Vector(
                        fen[f"vertex_{i}_x_coordinate"],
                        fen[f"vertex_{i}_y_coordinate"],
                        fen[f"vertex_{i}_z_coordinate"],
                    )
                )
            polygon = radgeom.Polygon(vertices)
            construction = fen['construction_name']
            fenestrations[name] = EPlusFenestration(
                name, surface_type, polygon, construction, host)
    return fenestrations


def parse_epjson(epjs: dict) -> tuple:
    """
    Convert EnergyPlus JSON objects into Radiance primitives.
    """
    # get site information
    site = list(epjs['Site:Location'].values())[0]

    # parse each fenestration
    fenestrations = parse_epjson_fenestration(epjs["FenestrationSurface:Detailed"])

    # Get all the fenestration hosting surfaces
    fene_hosts = set([val.host for val in fenestrations.values()])

    # parse each opaque surface
    opaque_surfaces = parse_opaque_surface(
        epjs["BuildingSurface:Detailed"], fenestrations)

    # parse each construction
    constructions = parse_construction(epjs['Construction'])

    cfs, matrices = parse_construction_complexfenestrationstate(epjs)
    constructions.update(cfs)

    # parse materials
    materials = parse_epjson_material(epjs)

    # get exterior zones
    exterior_zones = [value.zone for key, value in opaque_surfaces.items()
                      if (key in fene_hosts) and value.sun_exposed]

    # get secondary zones, but we don't do anything with it yet.
    secondary_zones = {}
    for key, value in opaque_surfaces.items():
        if (key in fene_hosts) and (value.zone not in exterior_zones):
            adjacent_zone = opaque_surfaces[value.boundary].zone
            if adjacent_zone in exterior_zones:
                secondary_zones[value.zone] = {}

    zones = {}
    # go through each exterior zone, update zone dictionary.
    for zname in exterior_zones:
        zone_name = zname.replace(" ", "_")
        surface_map = {"Wall": {}, "Ceiling": {}, "Roof": {}, "Floor": {}}
        windows = {n: val for n, val in fenestrations.items()
                   if opaque_surfaces[val.host].zone == zname}
        for name, surface in opaque_surfaces.items():
            if surface.zone == zname:
                surface_map[surface.type][name] = surface
        zones[zname] = EPlusZone(zone_name, *surface_map.values(), windows)
    return site, zones, constructions, materials, matrices


def write_config(config):
    cfg = ConfigParser(allow_no_value=True)
    templ_config = config.to_dict()
    cfg.read_dict(templ_config)
    with open(f"{config.name}.cfg", "w") as rdr:
        cfg.write(rdr)


def epjson2rad(epjs: dict) -> dict:
    """Command-line program to convert a energyplus model into a Radiance model."""
    # Setup file structure
    util.mkdir_p("Objects")
    util.mkdir_p("Resources")
    util.mkdir_p("Matrices")

    site, zones, constructions, materials, matrices = parse_epjson(epjs)
    building_name = epjs["Building"].popitem()[0].replace(" ", "_")

    # Write material file
    material_name = f"materials{building_name}.mat"
    with open(os.path.join("Objects", material_name), 'w') as wtr:
        for material in materials.values():
            wtr.write(str(material.primitive))

    # Write matrix files to xml, if any
    xml_paths = {}
    for key, val in matrices.items():
        opath = os.path.join('Resources', key + '.xml')
        tf_path = os.path.join('Resources', key + '_tf.mtx')
        tb_path = os.path.join('Resources', key + '_tb.mtx')
        with open(tf_path, 'w') as wtr:
            wtr.write(repr(val.tf))
        with open(tb_path, 'w') as wtr:
            wtr.write(repr(val.tb))
        basis = ''.join([word[0] for word in val.tf.basis.split()])
        cmd = ['wrapBSDF', '-f', 'n=' + key, '-a', basis]
        cmd += ['-tf', tf_path, '-tb', tb_path, '-U']
        wb_process = sp.run(cmd, check=True, stdout=sp.PIPE, stderr=sp.PIPE)
        with open(opath, 'wb') as wtr:
            wtr.write(wb_process.stdout)
        xml_paths[key] = opath

    zone_config = {}
    # For each zone write primitves to files and create a config file
    for name, zone in zones.items():
        mrad_config = util.MradConfig(
            latitude=site["latitude"],
            longitude=site["longitude"],
            material=material_name,
        )
        primitives = epluszone2rad(zone, constructions, materials)
        scene = []
        windows = []
        window_xmls = []
        window_controls = []
        floors = []
        for primitive in primitives:
            write_primitives(primitive, "Objects")
            for _name, item in primitive.items():
                if item["surface"] != []:
                    scene.append(_name + ".rad")
                if item["window"] != []:
                    windows.append(_name + "_window.rad")
                if item["xml"] != []:
                    window_xmls.extend(item['xml'])
                    window_controls.append("0")
        # Get floors
        for primitive in primitives[-1]:
            floors.append(primitive + ".rad")

        mrad_config.scene = " ".join(scene)
        mrad_config.window_paths = " ".join(windows)
        mrad_config.window_xml = " ".join(window_xmls)
        mrad_config.window_control = " ".join(window_controls)
        mrad_config.grid_surface = " ".join(floors)
        mrad_config.name = name
        mrad_config.__post_init__()
        zone_config[name] = mrad_config
    return zone_config


def read_ep_input(fpath: str) -> dict:
    """Load and parse input file into a JSON object.
    If the input file is in .idf fomart, use command-line
    energyplus program to convert it to epJSON format
    Args:
        fpath: input file path
    Returns:
        epjs: JSON object as a Python dictionary
    """
    epjson_path: str = ''
    if fpath.endswith('.idf'):
        cmd = ['energyplus', '--convert-only', fpath]
        sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        epjson_path = os.path.splitext(os.path.basename(fpath))[0] + '.epJSON'
        if not os.path.isfile(epjson_path):
            raise OSError("idf to epjson conversion failed")
    elif fpath.endswith('.epJSON'):
        epjson_path = fpath
    with open(epjson_path) as rdr:
        epjs = json.load(rdr)
    return epjs


def epjson2rad_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath')
    parser.add_argument('-run', action='store_true', default=False)
    args = parser.parse_args()
    epjs = read_ep_input(args.fpath)
    if "FenestrationSurface:Detailed" not in epjs:
        raise ValueError("No windows found in this model")
    configs = epjson2rad(epjs)
    for config in configs.values():
        write_config(config)
