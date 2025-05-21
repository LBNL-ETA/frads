import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from enum import Enum, auto

import math
import numpy as np
from frads.geom import polygon_primitive, Polygon, angle_between
import pyradiance as pr
import pywincalc as pwc

AIR = pwc.PredefinedGasType.AIR
KRYPTON = pwc.PredefinedGasType.KRYPTON
XENON = pwc.PredefinedGasType.XENON
ARGON = pwc.PredefinedGasType.ARGON

M_PER_MM = 0.001
NM_PER_MM = 1e3


class WCProductType(Enum):
    SHADING = auto()
    GLAZING = auto()


class WCProductSubType(Enum):
    VENETIAN = auto()
    FABRIC = auto()


@dataclass(slots=True)
class PaneRGB:
    """Pane color data object.

    Attributes:
        measured_data: measured data as a PaneProperty object.
        coated_rgb: Coated side RGB.
        glass_rgb: Non-coated side RGB.
        trans_rgb: Transmittance RGB.
    """

    coated_rgb: tuple[float, float, float]
    glass_rgb: tuple[float, float, float]
    trans_rgb: tuple[float, float, float]
    coated_side: None | str = None

@dataclass(slots=True)
class OpeningDistances:
    top_opening_distance_m: float = 0.0
    bottom_opening_distance_m: float = 0.0
    left_opening_distance_m: float = 0.0
    right_opening_distance_m: float = 0.0

@dataclass(slots=True)
class OpeningMultipliers:
    top_opening_multiplier: float = 0.0
    bottom_opening_multiplier: float = 0.0
    left_opening_multiplier: float = 0.0
    right_opening_multiplier: float = 0.0
    front_opening_multiplier: float = 0.0

@dataclass(slots=True)
class BaseLayerDefinition:
    input_source: Path | str | bytes
    flipped: bool = False

@dataclass(slots=True)
class GlazingLayerDefinition(BaseLayerDefinition):
    pass

@dataclass(slots=True)
class BlindsLayerDefinition(BaseLayerDefinition):
    slat_angle_deg: float = 0.0
    openings: OpeningDistances = field(default_factory=OpeningDistances)

@dataclass(slots=True)
class FabricLayerDefinition(BaseLayerDefinition):
    openings: OpeningDistances = field(default_factory=OpeningDistances)

AnyLayerDefinition = GlazingLayerDefinition | BlindsLayerDefinition | FabricLayerDefinition

@dataclass(slots=True)
class Layer:
    """Layer data object.

    Attributes:
        product_name: Name of the product.
        thickness: Thickness of the layer.
        product_type: Type of product.
        conductivity: Conductivity of the layer.
        emissivity_front: Front emissivity of the layer.
        emissivity_back: Back emissivity of the layer.
        ir_transmittance: IR transmittance of the layer.
        rgb: PaneRGB object.
    """

    product_name: str
    thickness_m: float
    product_type: str
    conductivity: float
    emissivity_front: float
    emissivity_back: float
    ir_transmittance: float
    flipped: bool = False
    spectral_data: None | dict = None
    shading_material: None | pr.ShadingMaterial = None
    slat_width_m: float = 0.0160
    slat_spacing_m: float = 0.0120
    slat_thickness_m: float = 0.0006
    slat_curve_m: float = 0.0
    slat_angle_deg: float = 90.0
    slat_conductivity: float = 160.00
    nslats: int = 1
    openings: OpeningMultipliers = field(default_factory=OpeningMultipliers)
    front_opening_multiplier: float = 0.0
    shading_xml: None | bytes = None


@dataclass(slots=True)
class Gas:
    """Gas data object.

    Attributes:
        gas: Gas type.
        ratio: Gas ratio.
    """

    gas: str
    ratio: float

    def __post_init__(self):
        if self.ratio < 0 or self.ratio > 1:
            raise ValueError("Gas ratio must be between 0 and 1.")
        if self.gas.lower() not in ("air", "argon", "krypton", "xenon"):
            raise ValueError("Invalid gas type.")


@dataclass(slots=True)
class Gap:
    """Gap data object.

    Attributes:
        gas: List of Gas objects.
        thickness: Thickness of the gap.
    """

    gas: list[Gas]
    thickness: float

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError("Gap thickness must be greater than 0.")
        if sum(g.ratio for g in self.gas) != 1:
            raise ValueError("The sum of the gas ratios must be 1.")


@dataclass(slots=True)
class GlazingSystem:
    """Glazing system data object.

    Attributes:
        name: Name of the glazing system.
        thickness: Thickness of the glazing system.
        layers: List of Layer objects.
        gaps: List of Gap objects.
        visible_front_transmittance: Visible front transmittance matrix.
        visible_back_transmittance: Visible back transmittance matrix.
        visible_front_reflectance: Visible front reflectance matrix.
        visible_back_reflectance: Visible back reflectance matrix.
        solar_front_transmittance: Solar front transmittance matrix.
        solar_back_transmittance: Solar back transmittance matrix.
        solar_front_reflectance: Solar front reflectance matrix.
        solar_back_reflectance: Solar back reflectance matrix.
        solar_front_absorptance: Solar front absorptance matrix by layer.
        solar_back_absorptance: Solar back absorptance matrix by layer.
    """

    name: str
    thickness: float = 0
    layers: list[Layer] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    visible_front_transmittance: list[list[float]] = field(default_factory=list)
    visible_back_transmittance: list[list[float]] = field(default_factory=list)
    visible_front_reflectance: list[list[float]] = field(default_factory=list)
    visible_back_reflectance: list[list[float]] = field(default_factory=list)
    solar_front_transmittance: list[list[float]] = field(default_factory=list)
    solar_back_transmittance: list[list[float]] = field(default_factory=list)
    solar_front_reflectance: list[list[float]] = field(default_factory=list)
    solar_back_reflectance: list[list[float]] = field(default_factory=list)
    solar_front_absorptance: list[list[float]] = field(default_factory=list)
    solar_back_absorptance: list[list[float]] = field(default_factory=list)
    melanopic_back_transmittance: list[list[float]] = field(default_factory=list)
    melanopic_back_reflectance: list[list[float]] = field(default_factory=list)

    def _matrix_to_str(self, matrix: list[list[float]]) -> str:
        """Convert a matrix to a string."""
        return "\n".join([" ".join([str(n) for n in row]) for row in matrix])

    def to_xml(self, out):
        """Save the glazing system to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _tmpdir = Path(tmpdir)
            with open(_tvf := _tmpdir / "tbv", "w") as f:
                f.write(self._matrix_to_str(self.visible_front_transmittance))
            with open(_tvb := _tmpdir / "tfv", "w") as f:
                f.write(self._matrix_to_str(self.visible_back_transmittance))
            with open(_rvf := _tmpdir / "rfv", "w") as f:
                f.write(self._matrix_to_str(self.visible_front_reflectance))
            with open(_rvb := _tmpdir / "rbv", "w") as f:
                f.write(self._matrix_to_str(self.visible_back_reflectance))
            with open(_tsf := _tmpdir / "tbs", "w") as f:
                f.write(self._matrix_to_str(self.solar_front_transmittance))
            with open(_tsb := _tmpdir / "tfs", "w") as f:
                f.write(self._matrix_to_str(self.solar_back_transmittance))
            with open(_rsf := _tmpdir / "rfs", "w") as f:
                f.write(self._matrix_to_str(self.solar_front_reflectance))
            with open(_rsb := _tmpdir / "rbs", "w") as f:
                f.write(self._matrix_to_str(self.solar_back_reflectance))
            with open(out, "wb") as f:
                f.write(
                    pr.WrapBSDF(
                        enforce_window=True,
                        basis="kf",
                        n=self.name,
                        m="",
                        t=str(self.thickness),
                    )
                    .add_visible(_tvb, _tvf, _rvb, _rvf)
                    .add_solar(_tsb, _tsf, _rsb, _rsf)()
                )

    def save(self, out: str | Path):
        """Save the glazing system to a file."""
        out = Path(out)
        if out.suffix == ".xml":
            self.to_xml(out)
        else:
            with open(out.with_suffix(".json"), "w") as f:
                json.dump(asdict(self), f)

    @classmethod
    def from_json(cls, path: str | Path):
        """Load a glazing system from a JSON file."""
        print("To be deprected please use load_glazing_system(fpath) instead.")
        with open(path, "r") as f:
            data = json.load(f)
        layers = data.pop("layers")
        layer_instances = []
        for layer in layers:
            smat = layer.pop("shading_material")
            shading_material = pr.ShadingMaterial(**smat) if smat is not None else None
            layer_instances.append(Layer(
                product_name=layer["product_name"],
                thickness_m=layer["thickness_m"],
                product_type=layer["product_type"],
                conductivity=layer["conductivity"],
                emissivity_front=layer["emissivity_front"],
                emissivity_back=layer["emissivity_back"],
                ir_transmittance=layer["ir_transmittance"],
                shading_material=shading_material,
            ))
        gaps = data.pop("gaps")
        gap_instances = []
        for gap in gaps:
            gas_instances = []
            gases = gap.pop("gas")
            for gs in gases:
                gas_instances.append(Gas(gas=gs["gas"], ratio=gs["ratio"]))
            gap_instances.append(Gap(gas=gas_instances, thickness=gap["thickness"], **gap))
        return cls(layers=layer_instances, gaps=gap_instances, **data)

    def get_brtdfunc(self, name: None | str = None) -> None:
        """Get a BRTDfunc primitive for the glazing system."""
        print("Deprecated: use get get_glazing_brtdfunc() instead.")
        return
        # if name is None:
        #     name = self.name
        # rgb = [layer.rgb for layer in self.layers if layer.rgb is not None]
        # return get_glazing_primitive(name, rgb)


def get_layer_data(inp: pwc.ProductData) -> Layer:
    """Create a list of layers from a list of pwc.ProductData."""
    return Layer(
        product_name=inp.product_name,
        thickness_m=inp.thickness,
        product_type=inp.product_type,
        conductivity=inp.conductivity,
        emissivity_front=inp.emissivity_front,
        emissivity_back=inp.emissivity_back,
        ir_transmittance=inp.ir_transmittance,
    )


def generate_blinds_xml(
    depth: float,
    nslats: int,
    angle: float,
    curvature: float,
    thickness: float,
    name: str,
    r_sol_diff: float,
    r_sol_spec: float,
    r_vis_diff: float,
    r_vis_spec: float,
    r_ir: float,
    nproc: int,
    nsamp: int,
) -> bytes:
    material_solar = pr.ShadingMaterial(r_sol_diff, r_sol_spec, 0)
    material_visible = pr.ShadingMaterial(r_vis_diff, r_vis_spec, 0)
    material_ir = pr.ShadingMaterial(r_ir, 0, 0)
    geom = pr.BlindsGeometry(
        depth=depth,
        width=1,
        height=1,
        nslats=nslats,
        angle=angle,
        rcurv=curvature,
    )
    sol_blinds = pr.generate_blinds_for_bsdf(material_solar, geom)
    vis_blinds = pr.generate_blinds_for_bsdf(material_visible, geom)
    ir_blinds = pr.generate_blinds_for_bsdf(material_ir, geom)
    sol_results = pr.generate_bsdf(sol_blinds, nproc=nproc, nsamp=nsamp)
    vis_results = pr.generate_bsdf(vis_blinds, nproc=nproc, nsamp=nsamp)
    ir_results = pr.generate_bsdf(ir_blinds, basis="u")
    return pr.generate_xml(
        sol_results,
        vis_results,
        ir_results,
        n=name,
        m="unnamed",
        t=thickness,
        tir=float(ir_results.front.transmittance.decode()),
    )


def load_glazing_system(path: str | Path) -> GlazingSystem:
    """Load a glazing system from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    layers = data.pop("layers")
    layer_instances = []
    for layer in layers:
        smat = layer.pop("shading_material")
        shading_material = pr.ShadingMaterial(**smat) if smat is not None else None
        layer_instances.append(Layer(
            product_name=layer["product_name"],
            thickness_m=layer["thickness_m"],
            product_type=layer["product_type"],
            conductivity=layer["conductivity"],
            emissivity_front=layer["emissivity_front"],
            emissivity_back=layer["emissivity_back"],
            ir_transmittance=layer["ir_transmittance"],
            shading_material=shading_material,
        ))
    gaps = data.pop("gaps")
    gap_instances = []
    for gap in gaps:
        gas_instances = []
        gases = gap.pop("gas")
        for gs in gases:
            gas_instances.append(Gas(gas=gs["gas"], ratio=gs["ratio"]))
        gap_instances.append(Gap(gas=gas_instances, thickness=gap["thickness"], **gap))
    return GlazingSystem(name=data["name"], layers=layer_instances, gaps=gap_instances, **data)


def create_pwc_gaps(gaps: list[Gap]):
    """Create a list of pwc gaps from a list of gaps."""
    pwc_gaps = []
    for gap in gaps:
        _gas = pwc.create_gas(
            [[g.ratio, getattr(pwc.PredefinedGasType, g.gas.upper())] for g in gap.gas]
        )
        _gap = pwc.Layers.gap(gas=_gas, thickness=gap.thickness)
        pwc_gaps.append(_gap)
    return pwc_gaps


def _parse_input_source(
    input_source: Path | str | bytes,
    layer_type_hint: str | None = None,
) -> pwc.ProductData:
    """Parses various input types to pwc.ProductData."""
    # Combines logic from original create_glazing_system for parsing json, xml, optics files, or bytes
    if isinstance(input_source, (str, Path)): #
        path = Path(input_source)
        if not path.exists():
            raise FileNotFoundError(f"{input_source} does not exist.") #
        if path.suffix == ".json":
            return pwc.parse_json_file(path) #
        elif path.suffix == ".xml": #
            # May need to distinguish if this XML is for fabric or already a complete blind BSDF
            return pwc.parse_bsdf_xml_file(path) #
        else:
            return pwc.parse_optics_file(path) #
    elif isinstance(input_source, bytes): #
        try:
            return pwc.parse_json(input_source) #
        except json.JSONDecodeError: #
            # This could be BSDF XML bytes
            return pwc.parse_bsdf_xml_string(input_source) #
    raise ValueError(f"Unsupported input_source type: {type(input_source)}")

def _apply_opening_properties(
    layer_obj: Layer,
    layer_def_openings: OpeningDistances | None,
    gaps_list: list[Gap],
    layer_idx: int,
    total_layers_in_definition: int
):
    """Calculates and sets opening multipliers on the layer object."""
    if layer_def_openings is None:
        return # No opening definitions to apply

    # Determine relevant gap thickness (simplified logic from original)
    # This logic needs to be robust based on layer position and gap availability.
    gap_thickness = 0.0127 # Default or average, replace with actual adjacent gap logic
    if total_layers_in_definition <= 1: # Single layer, no gaps to measure against effectively for openings
        pass # Or handle as fully open/closed based on convention
    elif layer_idx == 0: # First layer
        if gaps_list: gap_thickness = gaps_list[0].thickness
    elif layer_idx == total_layers_in_definition -1: # Last layer
        if gaps_list: gap_thickness = gaps_list[layer_idx-1].thickness
    else: # Middle layer
        if gaps_list and len(gaps_list) > layer_idx : # Check if index is valid
             gap_thickness = min(gaps_list[layer_idx - 1].thickness, gaps_list[layer_idx].thickness) #

    if gap_thickness > 0:
        layer_obj.openings.top_opening_multiplier = min(1., layer_def_openings.top_opening_distance_m / gap_thickness)
        layer_obj.openings.bottom_opening_multiplier = min(1., layer_def_openings.bottom_opening_distance_m / gap_thickness)
        layer_obj.openings.left_opening_multiplier = min(1., layer_def_openings.left_opening_distance_m / gap_thickness)
        layer_obj.openings.right_opening_multiplier = min(1., layer_def_openings.right_opening_distance_m / gap_thickness)
    # layer_obj.openings.front_opening_multiplier = layer_def_openings.front_opening_multiplier #


def _process_blind_definition_to_bsdf(
    defin: BlindsLayerDefinition,
    product_data: pwc.ProductData,
    nproc: int,
    nsamp: int
) -> tuple[pwc.ProductData, dict]:
    """
    Handles the logic for generating BSDF for blinds if defined by geometry/material.
    Returns the pwc.ProductData of the generated BSDF and a dict of blind properties.
    """
    if isinstance(defin.input_source, bytes):
        data = json.loads(defin.input_source.decode())
    else:
        with open(defin.input_source, 'r') as f:
            data = json.load(f.read())
    dual_band_values = data["composition"][0]["child_product"]["spectral_data"]["dual_band_values"]
    rf_sol_diff = dual_band_values["Rf_sol_diffuse"]
    rf_sol_spec = dual_band_values["Rf_sol_specular"]
    rf_vis_diff = dual_band_values["Rf_vis_diffuse"]
    rf_vis_spec = dual_band_values["Rf_vis_specular"]
    slat_spacing_m = product_data.composition.geometry.slat_spacing * M_PER_MM
    tir = product_data.composition.material.ir_transmittance
    nslats = int(1 / slat_spacing_m) if slat_spacing_m > 0 else 1
    slat_depth_m = product_data.composition.geometry.slat_width * M_PER_MM
    slat_curvature_m = product_data.composition.geometry.slat_curvature * M_PER_MM
    emis_front = product_data.composition.material.emissivity_front #
    emis_back = product_data.composition.material.emissivity_back #
    layer_name = product_data.product_name
    layer_thickness_m = slat_depth_m * math.cos(
        math.radians(defin.slat_angle_deg)
    )
    if tir != 0:
        raise ValueError("tir not zero")
    if emis_front != emis_back:
        raise ValueError("front and back emissivity not the same")
    rf_ir = 1 - tir - emis_front

    xml = generate_blinds_xml(
        slat_depth_m,
        nslats,
        defin.slat_angle_deg,
        slat_curvature_m,
        layer_thickness_m,
        layer_name,
        rf_sol_diff,
        rf_sol_spec,
        rf_vis_diff,
        rf_vis_spec,
        rf_ir,
        nproc,
        nsamp,
    )  # meters
    actual_product_data = pwc.parse_bsdf_xml_string(xml)
    # layer = get_layer_data(actual_product_data)
    # layer.slat_width_m = slat_depth_m
    # layer.slat_spacing_m = slat_spacing_m
    # layer.slat_thickness_m = product_data.composition.material.thickness * M_PER_MM
    # layer.slat_conductivity = product_data.composition.material.conductivity
    # layer.slat_curve_m = slat_curvature_m
    # layer.flipped = defin.flipped
    # layer.product_type = "blinds"
    # layer.shading_material = pr.ShadingMaterial(rf_vis_diff, rf_vis_spec, 0)
    # layer.nslats = nslats

    # layer.slat_angle_deg = defin.slat_angle_deg


    blind_props = {
        "slat_width_m": slat_depth_m,
        "slat_spacing_m": slat_spacing_m,
        "slat_thickness_m": product_data.composition.material.thickness * M_PER_MM,
        "slat_angle_deg": defin.slat_angle_deg,
        "slat_conductivity": product_data.composition.material.conductivity, #
        "slat_curve_m": slat_curvature_m,
        "nslats": nslats,
        "shading_material_rad": pr.ShadingMaterial(rf_vis_diff, rf_vis_spec, 0)
    }
    return actual_product_data, blind_props


def create_glazing_system(
    name: str,
    layer_inputs: list[AnyLayerDefinition],
    gaps: None | list[Gap] = None,
    nproc: int = 1,
    nsamp: int = 2000,
    mbsdf: bool = False,
) -> GlazingSystem:
    """Create a glazing system from a list of layers and gaps.

    Args:
        name: Name of the glazing system.
        layers: List of layers.
        gaps: List of gaps.

    Returns:
        GlazingSystem object.

    Raises:
        ValueError: Invalid layer type.

    Examples:
        >>> create_glazing_system(
        ...     "test",
        ...     [
        ...         Path("glass.json"),
        ...         Path("venetian.xml"),
        ...     ],
        ... )
    """
    if gaps is None:
        gaps = [Gap([Gas("air", 1)], 0.0127) for _ in range(len(layer_inputs) - 1)]
    product_data_list: list[pwc.ProductData] = []
    layer_data: list[Layer] = []
    thickness = 0.0
    fabric_xml: None | bytes
    for idx, layer_inp in enumerate(layer_inputs):
        product_data = _parse_input_source(layer_inp.input_source)
        if product_data is None:
            raise ValueError("Invalid layer type")
        layer = get_layer_data(product_data)
        if product_data.product_type == WCProductType.SHADING:
            if product_data.product_subtype == WCProductSubType.VENETIAN:
                actual_product_data, props = _process_blind_definition_to_bsdf(layer_inp, product_data, nproc= nproc, nsamp = nsamp)
                layer = Layer(**props)
                layer_data.append(layer)
                product_data_list.append(actual_product_data)
                thickness += layer.thickness_m
            else:
                layer.product_type = "fabric"
                with open(layer_inp.input_source, 'r') as f:
                    layer.shading_xml = f.read()
                layer_data.append(layer)
                product_data_list.append(product_data)
                thickness += layer.thickness_m
        else:
            layer = get_layer_data(product_data)
            layer.product_type = "glazing"
            layer.spectral_data = {
                int(round(d.wavelength * NM_PER_MM)): (
                    d.direct_component.transmittance_front,
                    d.direct_component.reflectance_front,
                    d.direct_component.reflectance_back,
                ) for d in product_data.measurements}
            layer_data.append(layer)
            product_data_list.append(product_data)
            thickness += layer.thickness_m

    for gap in gaps:
        thickness += gap.thickness

    glzsys = pwc.GlazingSystem(
        solid_layers=product_data_list,
        gap_layers=create_pwc_gaps(gaps),
        width_meters=1,
        height_meters=1,
        environment=pwc.nfrc_shgc_environments(),
        bsdf_hemisphere=pwc.BSDFHemisphere.create(pwc.BSDFBasisType.FULL),
    )

    if mbsdf:
        generate_melanopic_bsdf(layer_inp)

    for index, data in enumerate(layer_data):
        if data.flipped:
            glzsys.flip_layer(index, True)

    solres = glzsys.optical_method_results("SOLAR")
    solsys = solres.system_results
    visres = glzsys.optical_method_results("PHOTOPIC")
    vissys = visres.system_results

    return GlazingSystem(
        name=name,
        thickness=thickness,
        layers=layer_data,
        gaps=gaps,
        solar_front_absorptance=[
            alpha.front.absorptance.angular_total for alpha in solres.layer_results
        ],
        solar_back_absorptance=[
            alpha.back.absorptance.angular_total for alpha in solres.layer_results
        ],
        visible_back_reflectance=vissys.back.reflectance.matrix,
        visible_front_reflectance=vissys.front.reflectance.matrix,
        visible_back_transmittance=vissys.back.transmittance.matrix,
        visible_front_transmittance=vissys.front.transmittance.matrix,
        solar_back_reflectance=solsys.back.reflectance.matrix,
        solar_front_reflectance=solsys.front.reflectance.matrix,
        solar_back_transmittance=solsys.back.transmittance.matrix,
        solar_front_transmittance=solsys.front.transmittance.matrix,
    )

def get_glazing_brtdfunc(name, filenames: list[str | Path]) -> pr.Primitive:
    nlayers = len(filenames)
    if nlayers > 3 or nlayers == 0:
        raise ValueError(f"Expected 1 to 3 layers, got {nlayers}.")
    panes_rgb: list[PaneRGB] = []
    for filename in filenames:
        fpath = Path(filename)
        if fpath.suffix == ".json":
            product_data = pwc.parse_json_files(str(filename))
        else:
            product_data = pwc.parse_optics_file(str(filename))
        rgb = get_layer_rgb(product_data)
        panes_rgb.append(rgb)
    return get_glazing_primitive(name, panes_rgb)


def get_layer_rgb(layer: pwc.ProductData) -> None | PaneRGB:
    """Get the RGB values for a pane layer."""
    photopic_wvl = range(380, 781, 10)
    if isinstance(layer.measurements, pwc.DualBandBSDF):
        return None
    hemi = {
        int(round(d.wavelength * NM_PER_MM)): (
            d.direct_component.transmittance_front,
            d.direct_component.transmittance_back,
            d.direct_component.reflectance_front,
            d.direct_component.reflectance_back,
        )
        for d in layer.measurements
    }
    tvf = [hemi[w][0] for w in photopic_wvl]
    rvf = [hemi[w][2] for w in photopic_wvl]
    rvb = [hemi[w][3] for w in photopic_wvl]
    tf_x, tf_y, tf_z = pr.spec_xyz(tvf, 380, 780)
    rf_x, rf_y, rf_z = pr.spec_xyz(rvf, 380, 780)
    rb_x, rb_y, rb_z = pr.spec_xyz(rvb, 380, 780)
    tf_rgb = pr.xyz_rgb(tf_x, tf_y, tf_z)
    rf_rgb = pr.xyz_rgb(rf_x, rf_y, rf_z)
    rb_rgb = pr.xyz_rgb(rb_x, rb_y, rb_z)
    if layer.coated_side == "front":
        coated_rgb = rf_rgb
        glass_rgb = rb_rgb
    else:
        coated_rgb = rb_rgb
        glass_rgb = rf_rgb
    return PaneRGB(coated_rgb, glass_rgb, tf_rgb, layer.coated_side)


def get_glazing_primitive(name: str, panes: list[PaneRGB]) -> pr.Primitive:
    """Generate a BRTDfunc to represent a glazing system."""
    if len(panes) == 1:
        str_arg = [
            "sr_clear_r",
            "sr_clear_g",
            "sr_clear_b",
            "st_clear_r",
            "st_clear_g",
            "st_clear_b",
            "0",
            "0",
            "0",
            "glaze1.cal",
        ]
        coated_real = 1 if panes[0].coated_side == "front" else -1
        real_arg = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            coated_real,
            *[round(i, 3) for i in panes[0].glass_rgb],
            *[round(i, 3) for i in panes[0].coated_rgb],
            *[round(i, 3) for i in panes[0].trans_rgb],
        ]
    elif len(panes) == 2:
        s12t_r, s12t_g, s12t_b = panes[0].trans_rgb
        s34t_r, s34t_g, s34t_b = panes[1].trans_rgb
        if panes[0].coated_side == "back":
            s2r_r, s2r_g, s2r_b = panes[0].coated_rgb
            s1r_r, s1r_g, s1r_b = panes[0].glass_rgb
        else:  # front or neither side coated
            s2r_r, s2r_g, s2r_b = panes[0].glass_rgb
            s1r_r, s1r_g, s1r_b = panes[0].coated_rgb
        if panes[1].coated_side == "back":
            s4r_r, s4r_g, s4r_b = panes[1].coated_rgb
            s3r_r, s3r_g, s3r_b = panes[1].glass_rgb
        else:  # front or neither side coated
            s4r_r, s4r_g, s4r_b = panes[1].glass_rgb
            s3r_r, s3r_g, s3r_b = panes[1].coated_rgb
        str_arg = [
            (
                f"if(Rdot,cr(fr({s4r_r:.3f}),ft({s34t_r:.3f}),fr({s2r_r:.3f})),"
                f"cr(fr({s1r_r:.3f}),ft({s12t_r:.3f}),fr({s3r_r:.3f})))"
            ),
            (
                f"if(Rdot,cr(fr({s4r_g:.3f}),ft({s34t_g:.3f}),fr({s2r_g:.3f})),"
                f"cr(fr({s1r_g:.3f}),ft({s12t_g:.3f}),fr({s3r_g:.3f})))"
            ),
            (
                f"if(Rdot,cr(fr({s4r_b:.3f}),ft({s34t_b:.3f}),fr({s2r_b:.3f})),"
                f"cr(fr({s1r_b:.3f}),ft({s12t_b:.3f}),fr({s3r_b:.3f})))"
            ),
            f"ft({s34t_r:.3f})*ft({s12t_r:.3f})",
            f"ft({s34t_g:.3f})*ft({s12t_g:.3f})",
            f"ft({s34t_b:.3f})*ft({s12t_b:.3f})",
            "0",
            "0",
            "0",
            "glaze2.cal",
        ]
        real_arg = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif len(panes) == 3:
        str_arg = [
            "sr_red",
            "sr_grn",
            "sr_blu",
            "st_red",
            "st_grn",
            "st_blu",
            "0",
            "0",
            "0",
            "glass3.cal",
        ]
        if panes[2].coated_side == "back":
            rbspc, gbspc, bbspc = panes[2].coated_rgb
        else:
            rbspc, gbspc, bbspc = panes[2].glass_rgb
        if panes[0].coated_side == "back":
            rfspc, gfspc, bfspc = panes[0].glass_rgb
        else:
            rfspc, gfspc, bfspc = panes[0].coated_rgb
        rtspc = panes[0].trans_rgb[0] * panes[1].trans_rgb[0] * panes[2].trans_rgb[0]
        gtspc = panes[0].trans_rgb[1] * panes[1].trans_rgb[1] * panes[2].trans_rgb[1]
        btspc = panes[0].trans_rgb[2] * panes[1].trans_rgb[2] * panes[2].trans_rgb[2]
        real_arg = [
            18,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            rfspc,
            gfspc,
            bfspc,
            rbspc,
            gbspc,
            bbspc,
            rtspc,
            gtspc,
            btspc,
        ]
    else:
        raise ValueError("At most three pane supported")
    return pr.Primitive("void", "BRTDfunc", name, str_arg, real_arg)


def get_glazing_layer_groups(layers: list[Layer]) -> list:
    if not layers or not isinstance(layers[0], Layer):
        raise ValueError()

    # Group consecutive glazing layers
    grouped_system: list[tuple[str, int]] = []
    current_group = 0
    current_type = None

    for layer in layers:
        if layer.product_type == "glazing":
            if current_type == "glazing":
                current_group += 1
            else:
                if current_group:
                    grouped_system.append((current_type, current_group))
                current_group = 1
                current_type = "glazing"
        else:
            if current_group:
                grouped_system.append((current_type, current_group))
            grouped_system.append((layer.product_type, 1))
            current_group = 0
            current_type = None

    # Add the last group if it exists
    if current_group:
        grouped_system.append((current_type, current_group))

    return grouped_system


def get_proxy_geometry(window: Polygon, gs: GlazingSystem) -> list[pr.Primitive]:
    FEPS = 1e-5
    layer_groups = get_glazing_layer_groups(gs.layers)
    window_vertices = window.vertices
    primitives: list[pr.Primitive] = []
    current_index = 0
    for group in layer_groups:
        if group[0] == "glazing":
            glazing_layer_count = group[1]
            # Get BRTDFunc
            mat_name = f"mat_{gs.name}_glazing_{current_index}"
            geom_name = f"{gs.name}_glazing_{current_index}"
            rgb = [
                layer.spectra_data
                for layer in gs.layers[
                    current_index : current_index + glazing_layer_count
                ]
            ]
            mat: bytes = get_glazing_primitive(mat_name, rgb)
            geom = polygon_primitive(window, mat_name, geom_name)
            primitives.append(mat.bytes)
            primitives.append(geom.bytes)
            current_index += glazing_layer_count
        elif group[0] == "fabric":
            mat_name = f"mat_{gs.name}_fabric_{current_index}"
            geom_name = f"{gs.name}_fabric_{current_index}"
            xml_name = f"{gs.name}_fabric_{current_index}.xml"
            # Get aBSDF primitive
            with open(xml_name, "wb") as f:
                f.write(gs.layers[current_index].fabric_xml)
            mat: pr.Primitive = pr.Primitive(
                "void", "aBSDF", mat_name, [xml_name, "0", "0", "1", "."], []
            )
            geom = polygon_primitive(window, mat_name, geom_name)
            primitives.append(mat.bytes)
            primitives.append(geom.bytes)
            current_index += 1
        elif group[0] == "blinds":
            # NOTE: For blinds only
            zdiff1 = abs(window_vertices[1][2] - window_vertices[0][2])
            zdiff2 = abs(window_vertices[2][2] - window_vertices[1][2])
            dim1 = np.linalg.norm(window_vertices[1] - window_vertices[0])
            dim2 = np.linalg.norm(window_vertices[2] - window_vertices[1])
            if (dim1 <= FEPS) | (dim2 <= FEPS):
                print("One of the sides of the window polygon is zero")
            width = height = 0
            if zdiff1 <= FEPS:
                width = dim1
                height = dim2
            elif zdiff2 <= FEPS:
                width = dim2
                height = dim1
            else:
                print("Error: Skewed window not supported: ")
                print(window_vertices)
            blinds_layer = gs.layers[current_index]
            geom = pr.BlindsGeometry(
                depth=blinds_layer.slat_width,
                width=width,
                height=height,
                nslats=blinds_layer.nslats,
                angle=blinds_layer.slat_angle,
                rcurv=blinds_layer.slat_curve,
            )
            blinds: bytes = pr.generate_blinds(blinds_layer.shading_material, geom)
            xmin, xmax, ymin, ymax, zmin, zmax = pr.ot.getbbox(blinds)
            blinds_normal = np.array((1, 0, 0))
            rotatez_angle = angle_between(blinds_normal, window.normal, degree=True)
            rotated_blinds: bytes = pr.Xform(blinds).rotatez(rotatez_angle)()
            xmin, xmax, ymin, ymax, zmin, zmax = pr.ot.getbbox(rotated_blinds)
            rotated_blinds_centroid = np.array(
                ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
            )
            blinds_at_window = pr.Xform(rotated_blinds).translate(
                *(window.centroid - rotated_blinds_centroid)
            )()
            primitives.append(
                pr.Xform(blinds_at_window).translate(
                    *(window.normal * gs.layers[current_index].thickness)
                )()
            )
            current_index += 1
    return primitives


def get_spectral_multi_layer_optics(glazings: Sequence[Layer]):
    primitives: bytes = pr.genglaze_db([g for g in glazings], prefix=name)


def generate_melanopic_bsdf(layer_inputs: Sequence[Layers]):
    layer_groups = get_glazing_layer_groups(layers)
    for group in layer_groups:
        if group[0] == "glazing":
            glazing_layer_count = group[1]
            mat_name = f"mat_{gs.name}_glazing_{current_index}"
            geom_name = f"{gs.name}_glazing_{current_index}"
            get_spectral_multi_layer_optics(gs.layers[current_index: current_indx+glazing_layer_count])
            geom = polygon_primitive(window, mat_name, geom_name)
            primitives.append(mat.bytes)
            primitives.append(geom.bytes)
            current_index += glazing_layer_count
        elif group[0] == "fabric":
            mat_name = f"mat_{gs.name}_fabric_{current_index}"
            geom_name = f"{gs.name}_fabric_{current_index}"
            xml_name = f"{gs.name}_fabric_{current_index}.xml"
            # Get aBSDF primitive
            with open(xml_name, "wb") as f:
                f.write(gs.layers[current_index].fabric_xml)
            mat: pr.Primitive = pr.Primitive(
                "void", "aBSDF", mat_name, [xml_name, "0", "0", "1", "."], []
            )
            geom = polygon_primitive(window, mat_name, geom_name)
            primitives.append(mat.bytes)
            primitives.append(geom.bytes)
            current_index += 1
        elif group[0] == "blinds":
            # NOTE: For blinds only
            zdiff1 = abs(window_vertices[1][2] - window_vertices[0][2])
            zdiff2 = abs(window_vertices[2][2] - window_vertices[1][2])
            dim1 = np.linalg.norm(window_vertices[1] - window_vertices[0])
            dim2 = np.linalg.norm(window_vertices[2] - window_vertices[1])
            if (dim1 <= FEPS) | (dim2 <= FEPS):
                print("One of the sides of the window polygon is zero")
            width = height = 0
            if zdiff1 <= FEPS:
                width = dim1
                height = dim2
            elif zdiff2 <= FEPS:
                width = dim2
                height = dim1
            else:
                print("Error: Skewed window not supported: ")
                print(window_vertices)
            blinds_layer = gs.layers[current_index]
            geom = pr.BlindsGeometry(
                depth=blinds_layer.slat_width,
                width=width,
                height=height,
                nslats=blinds_layer.nslats,
                angle=blinds_layer.slat_angle,
                rcurv=blinds_layer.slat_curve,
            )
            blinds: bytes = pr.generate_blinds(blinds_layer.shading_material, geom)
            xmin, xmax, ymin, ymax, zmin, zmax = pr.ot.getbbox(blinds)
            blinds_normal = np.array((1, 0, 0))
            rotatez_angle = angle_between(blinds_normal, window.normal, degree=True)
            rotated_blinds: bytes = pr.Xform(blinds).rotatez(rotatez_angle)()
            xmin, xmax, ymin, ymax, zmin, zmax = pr.ot.getbbox(rotated_blinds)
            rotated_blinds_centroid = np.array(
                ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
            )
            blinds_at_window = pr.Xform(rotated_blinds).translate(
                *(window.centroid - rotated_blinds_centroid)
            )()
            primitives.append(
                pr.Xform(blinds_at_window).translate(
                    *(window.normal * gs.layers[current_index].thickness)
                )()
            )
            current_index += 1
    return primitives
