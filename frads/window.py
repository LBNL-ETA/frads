import json
import os
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pyradiance as pr
import pywincalc as pwc

from frads.geom import Polygon, angle_between, polygon_primitive
from frads.utils import parse_primitive

AIR = pwc.PredefinedGasType.AIR
KRYPTON = pwc.PredefinedGasType.KRYPTON
XENON = pwc.PredefinedGasType.XENON
ARGON = pwc.PredefinedGasType.ARGON

M_PER_MM = 0.001
NM_PER_MM = 1e3


SHADING = "shading"
GLAZING = "glazing"
VENETIAN = "venetian"
FABRIC = "fabric"


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
class OpeningDefinitions:
    top_m: float = 0.0
    bottom_m: float = 0.0
    left_m: float = 0.0
    right_m: float = 0.0
    front_multiplier: float = 0.05


@dataclass(slots=True)
class OpeningMultipliers:
    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0
    front: float = 0.0


@dataclass(slots=True)
class LayerInput:
    input_source: Path | str | bytes
    flipped: bool = False
    slat_angle_deg: float = 0.0
    openings: OpeningDefinitions = field(default_factory=OpeningDefinitions)


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
    spectral_data: dict = field(default_factory=dict)
    coated_side: str = ""
    shading_material: pr.ShadingMaterial = field(default_factory=pr.ShadingMaterial)
    slat_width_m: float = 0.0160
    slat_spacing_m: float = 0.0120
    slat_thickness_m: float = 0.0006
    slat_curve_m: float = 0.0
    slat_angle_deg: float = 90.0
    slat_conductivity: float = 160.00
    nslats: int = 1
    opening_multipliers: OpeningMultipliers = field(default_factory=OpeningMultipliers)
    shading_xml: str = ""


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
    thickness_m: float

    def __post_init__(self):
        if self.thickness_m <= 0:
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
                    .add_visible(str(_tvb), str(_tvf), str(_rvb), str(_rvf))
                    .add_solar(str(_tsb), str(_tsf), str(_rsb), str(_rsf))()
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
        print("\nTo be deprected please use load_glazing_system(fpath) instead.\n")
        with open(path, "r") as f:
            data = json.load(f)
        layers = data.pop("layers")
        layer_instances = []
        for layer in layers:
            smat = layer.pop("shading_material")
            shading_material = pr.ShadingMaterial(**smat)

            # Handle opening_multipliers nested dataclass
            opening_multipliers_data = layer.get("opening_multipliers", {})
            if isinstance(opening_multipliers_data, dict):
                opening_multipliers = OpeningMultipliers(**opening_multipliers_data)
            else:
                opening_multipliers = OpeningMultipliers()

            layer_instances.append(
                Layer(
                    product_name=layer["product_name"],
                    thickness_m=layer["thickness_m"],
                    product_type=layer["product_type"],
                    conductivity=layer["conductivity"],
                    emissivity_front=layer["emissivity_front"],
                    emissivity_back=layer["emissivity_back"],
                    ir_transmittance=layer["ir_transmittance"],
                    flipped=layer.get("flipped", False),
                    spectral_data={
                        int(k): v for k, v in layer["spectral_data"].items()
                    },
                    coated_side=layer.get("coated_side", ""),
                    shading_material=shading_material,
                    slat_width_m=layer.get("slat_width_m", 0.0160),
                    slat_spacing_m=layer.get("slat_spacing_m", 0.0120),
                    slat_thickness_m=layer.get("slat_thickness_m", 0.0006),
                    slat_curve_m=layer.get("slat_curve_m", 0.0),
                    slat_angle_deg=layer.get("slat_angle_deg", 90.0),
                    slat_conductivity=layer.get("slat_conductivity", 160.00),
                    nslats=layer.get("nslats", 1),
                    opening_multipliers=opening_multipliers,
                    shading_xml=layer.get("shading_xml", ""),
                )
            )
        gaps = data.pop("gaps")
        gap_instances = []
        for gap in gaps:
            gases = gap.pop("gas")
            gas_instances = [Gas(gas=gas["gas"], ratio=gas["ratio"]) for gas in gases]
            gap_instances.append(
                Gap(
                    gas=gas_instances,
                    thickness_m=gap["thickness_m"],
                    **{k: v for k, v in gap.items() if k not in ["gas", "thickness_m"]},
                )
            )
        return cls(layers=layer_instances, gaps=gap_instances, **data)

    def get_brtdfunc(self, name: None | str = None) -> None:
        """Get a BRTDfunc primitive for the glazing system."""
        print("Deprecated: use get get_glazing_brtdfunc() instead.")
        return


def get_layer_data(inp: pwc.ProductData) -> Layer:
    """Create a list of layers from a list of pwc.ProductData."""
    layer = Layer(
        product_name=inp.product_name,
        product_type=inp.product_type,
        thickness_m=inp.thickness,
        conductivity=inp.conductivity,
        emissivity_front=inp.emissivity_front,
        emissivity_back=inp.emissivity_back,
        ir_transmittance=inp.ir_transmittance,
    )
    if layer.thickness_m is not None:
        layer.thickness_m *= M_PER_MM
    return layer


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
    width = 1
    height = 1
    material_solar = pr.ShadingMaterial(r_sol_diff, r_sol_spec, 0)
    material_visible = pr.ShadingMaterial(r_vis_diff, r_vis_spec, 0)
    material_ir = pr.ShadingMaterial(r_ir, 0, 0)
    geom = pr.BlindsGeometry(
        depth=depth,
        width=width,
        height=height,
        nslats=nslats,
        angle=angle,
        rcurv=curvature,
    )
    sol_blinds = pr.generate_blinds_for_bsdf(material_solar, geom)
    vis_blinds = pr.generate_blinds_for_bsdf(material_visible, geom)
    ir_blinds = pr.generate_blinds_for_bsdf(material_ir, geom)
    spacing = height / nslats
    dim = pr.genbsdf.SamplingBox(0.5, 0.51, 0.5, 0.5 + spacing, -thickness, 0)
    sol_results = pr.generate_bsdf(sol_blinds, dim=dim, nproc=nproc, nsamp=nsamp)
    vis_results = pr.generate_bsdf(vis_blinds, dim=dim, nproc=nproc, nsamp=nsamp)
    ir_results = pr.generate_bsdf(ir_blinds, dim=dim, basis="u")
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
    """Load a glazing system from a JSON file.

    Args:
        path: Path to the JSON file containing glazing system definition.

    Returns:
        GlazingSystem object with loaded optical and thermal properties.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the JSON file format is invalid.

    Examples:
        >>> gs = load_glazing_system("my_window.json")
        >>> print(f"U-factor: {gs.u_factor}")
    """
    with open(path, "r") as f:
        data = json.load(f)
    layers = data.pop("layers")
    layer_instances = []
    for layer in layers:
        smat = layer.pop("shading_material")
        shading_material = pr.ShadingMaterial(**smat)

        # Handle opening_multipliers nested dataclass
        opening_multipliers_data = layer.get("opening_multipliers", {})
        if isinstance(opening_multipliers_data, dict):
            opening_multipliers = OpeningMultipliers(**opening_multipliers_data)
        else:
            opening_multipliers = OpeningMultipliers()

        layer_instances.append(
            Layer(
                product_name=layer["product_name"],
                thickness_m=layer["thickness_m"],
                product_type=layer["product_type"],
                conductivity=layer["conductivity"],
                emissivity_front=layer["emissivity_front"],
                emissivity_back=layer["emissivity_back"],
                ir_transmittance=layer["ir_transmittance"],
                flipped=layer.get("flipped", False),
                spectral_data={int(k): v for k, v in layer["spectral_data"].items()},
                coated_side=layer.get("coated_side", ""),
                shading_material=shading_material,
                slat_width_m=layer.get("slat_width_m", 0.0160),
                slat_spacing_m=layer.get("slat_spacing_m", 0.0120),
                slat_thickness_m=layer.get("slat_thickness_m", 0.0006),
                slat_curve_m=layer.get("slat_curve_m", 0.0),
                slat_angle_deg=layer.get("slat_angle_deg", 90.0),
                slat_conductivity=layer.get("slat_conductivity", 160.00),
                nslats=layer.get("nslats", 1),
                opening_multipliers=opening_multipliers,
                shading_xml=layer.get("shading_xml", ""),
            )
        )
    gaps = data.pop("gaps")
    gap_instances = []
    for gap in gaps:
        gases = gap.pop("gas")
        gas_instances = [Gas(gas=gas["gas"], ratio=gas["ratio"]) for gas in gases]
        gap_instances.append(
            Gap(
                gas=gas_instances,
                thickness_m=gap["thickness_m"],
                **{k: v for k, v in gap.items() if k not in ["gas", "thickness_m"]},
            )
        )
    return GlazingSystem(layers=layer_instances, gaps=gap_instances, **data)


def create_pwc_gaps(gaps: list[Gap]):
    """Create a list of pwc gaps from a list of gaps."""
    pwc_gaps = []
    for gap in gaps:
        _gas = pwc.create_gas(
            [[g.ratio, getattr(pwc.PredefinedGasType, g.gas.upper())] for g in gap.gas]
        )
        _gap = pwc.Layers.gap(gas=_gas, thickness=gap.thickness_m)
        pwc_gaps.append(_gap)
    return pwc_gaps


def _parse_input_source(
    input_source: Path | str | bytes,
) -> pwc.ProductData:
    """Parses various input types to pwc.ProductData."""
    if isinstance(input_source, (str, Path)):
        path = Path(input_source)
        if not path.exists():
            raise FileNotFoundError(f"{input_source} does not exist.")
        if path.suffix == ".json":
            return pwc.parse_json_file(str(path))
        elif path.suffix == ".xml":
            product_data = pwc.parse_bsdf_xml_file(str(path))
            if product_data.product_type == "" or product_data.product_type is None:
                product_data.product_type = SHADING
            return product_data
        else:
            return pwc.parse_optics_file(str(path))
    elif isinstance(input_source, bytes):
        try:
            return pwc.parse_json(input_source)
        except json.JSONDecodeError:
            product_data = pwc.parse_bsdf_xml_string(input_source.decode())
            if product_data.product_type == "" or product_data.product_type is None:
                product_data.product_type = SHADING
            return product_data


def _apply_opening_properties(
    layer_obj: Layer,
    layer_def_openings: OpeningDefinitions | None,
    gaps_list: list[Gap],
    layer_idx: int,
    total_layers_in_definition: int,
):
    """Calculates and sets opening multipliers on the layer object."""
    if layer_def_openings is None:
        return  # No opening definitions to apply

    gap_thickness = 0.0127
    if total_layers_in_definition <= 1:
        pass
    elif layer_idx == 0:
        if gaps_list:
            gap_thickness = gaps_list[0].thickness_m
    elif layer_idx == total_layers_in_definition - 1:
        if gaps_list:
            gap_thickness = gaps_list[layer_idx - 1].thickness_m
    else:  # Middle layer
        if gaps_list and len(gaps_list) > layer_idx:
            gap_thickness = min(
                gaps_list[layer_idx - 1].thickness_m, gaps_list[layer_idx].thickness_m
            )  #

    if gap_thickness > 0:
        layer_obj.opening_multipliers.top = min(
            1.0, layer_def_openings.top_m / gap_thickness
        )
        layer_obj.opening_multipliers.bottom = min(
            1.0, layer_def_openings.bottom_m / gap_thickness
        )
        layer_obj.opening_multipliers.left = min(
            1.0, layer_def_openings.left_m / gap_thickness
        )
        layer_obj.opening_multipliers.right = min(
            1.0, layer_def_openings.right_m / gap_thickness
        )
        layer_obj.opening_multipliers.front = min(
            1.0, layer_def_openings.front_multiplier
        )


def _process_blind_definition_to_bsdf(
    defin: LayerInput,
    product_data: pwc.ProductData,
    layer: Layer,
    nproc: int,
    nsamp: int,
) -> tuple[pwc.ProductData, dict]:
    """
    Handles the logic for generating BSDF for blinds if defined by geometry/material.
    Returns the pwc.ProductData of the generated BSDF and a dict of blind properties.
    """
    if isinstance(defin.input_source, bytes):
        data = json.loads(defin.input_source.decode())
    else:
        with open(defin.input_source, "r") as f:
            data = json.load(f)
    dual_band_values = data["composition"][0]["child_product"]["spectral_data"][
        "dual_band_values"
    ]
    rf_sol_diff = dual_band_values["Rf_sol_diffuse"]
    rf_sol_spec = dual_band_values["Rf_sol_specular"]
    rf_vis_diff = dual_band_values["Rf_vis_diffuse"]
    rf_vis_spec = dual_band_values["Rf_vis_specular"]
    geometry_def = product_data.geometry
    material_def = product_data.material_definition
    slat_spacing_m = geometry_def.slat_spacing * M_PER_MM
    tir = material_def.ir_transmittance
    nslats = int(1 / slat_spacing_m) if slat_spacing_m > 0 else 1
    slat_depth_m = geometry_def.slat_width * M_PER_MM
    slat_curvature_m = geometry_def.slat_curvature * M_PER_MM
    emis_front = material_def.emissivity_front  #
    emis_back = material_def.emissivity_back  #
    layer_name = product_data.product_name
    layer_thickness_m = slat_depth_m * math.cos(math.radians(defin.slat_angle_deg))
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
    layer.thickness_m = layer_thickness_m
    layer.conductivity = product_data.material_definition.conductivity
    layer.emissivity_front = emis_front
    layer.emissivity_back = emis_back
    layer.ir_transmittance = tir
    layer.product_type = "blinds"
    layer.slat_width_m = slat_depth_m
    layer.slat_spacing_m = slat_spacing_m
    layer.slat_thickness_m = product_data.material_definition.thickness * M_PER_MM
    layer.slat_conductivity = product_data.material_definition.conductivity
    layer.slat_curve_m = slat_curvature_m
    layer.flipped = defin.flipped
    layer.shading_material = pr.ShadingMaterial(rf_vis_diff, rf_vis_spec, 0)
    layer.nslats = nslats
    layer.slat_angle_deg = defin.slat_angle_deg

    return actual_product_data


def create_glazing_system(
    name: str,
    layer_inputs: list[LayerInput],
    gaps: None | list[Gap] = None,
    nproc: int = 1,
    nsamp: int = 2000,
    mbsdf: bool = False,
) -> GlazingSystem:
    """Create a glazing system from a list of layers and gaps.

    Args:
        name: Name of the glazing system.
        layer_inputs: List of layer inputs containing material specifications.
        gaps: List of gaps between layers (auto-generated if None).
        nproc: Number of processes for parallel computation.
        nsamp: Number of samples for Monte Carlo integration.
        mbsdf: Whether to generate melanopic BSDF data.

    Returns:
        GlazingSystem object containing optical and thermal properties.

    Raises:
        ValueError: Invalid layer type or input format.

    Examples:
        >>> from frads import LayerInput
        >>> layers = [
        ...     LayerInput("glass.json"),
        ...     LayerInput("venetian.xml")
        ... ]
        >>> gs = create_glazing_system("double_glazed", layers)
    """
    if gaps is None:
        gaps = [Gap([Gas("air", 1)], 0.0127) for _ in range(len(layer_inputs) - 1)]
    product_data_list: list[pwc.ProductData] = []
    layer_data: list[Layer] = []
    thickness = 0.0
    for idx, layer_inp in enumerate(layer_inputs):
        product_data = _parse_input_source(layer_inp.input_source)
        if product_data is None:
            raise ValueError("Invalid layer type")
        layer = get_layer_data(product_data)
        if product_data.product_type == SHADING:
            if product_data.product_subtype == VENETIAN:
                actual_product_data = _process_blind_definition_to_bsdf(
                    layer_inp, product_data, layer, nproc=nproc, nsamp=nsamp
                )
                layer_data.append(layer)
                product_data_list.append(actual_product_data)
                thickness += layer.thickness_m
            else:
                layer.product_type = FABRIC
                with open(layer_inp.input_source, "r") as f:
                    layer.shading_xml = f.read()
                layer_data.append(layer)
                product_data_list.append(product_data)
                thickness += layer.thickness_m
            _apply_opening_properties(
                layer, layer_inp.openings, gaps, idx, len(layer_inputs)
            )
        else:
            layer.spectral_data = {
                int(round(d.wavelength * NM_PER_MM)): (
                    d.direct_component.transmittance_front,
                    d.direct_component.reflectance_front,
                    d.direct_component.reflectance_back,
                )
                for d in product_data.measurements
            }
            layer.coated_side = product_data.coated_side
            layer_data.append(layer)
            product_data_list.append(product_data)
            thickness += layer.thickness_m

    for gap in gaps:
        thickness += gap.thickness_m

    glzsys = pwc.GlazingSystem(
        solid_layers=product_data_list,
        gap_layers=create_pwc_gaps(gaps),
        width_meters=1,
        height_meters=1,
        environment=pwc.nfrc_shgc_environments(),
        bsdf_hemisphere=pwc.BSDFHemisphere.create(pwc.BSDFBasisType.FULL),
    )

    melanopic_back_transmittace = []
    melanopic_back_reflectance = []
    if mbsdf:
        melanopic_back_transmittace, melanopic_back_reflectance = (
            generate_melanopic_bsdf(layer_data, gaps, nproc=nproc, nsamp=nsamp)
        )

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
        melanopic_back_transmittance=melanopic_back_transmittace,
        melanopic_back_reflectance=melanopic_back_reflectance,
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


def get_pane_rgb(spectral_data: dict, coated_side: str) -> PaneRGB:
    photopic_wvl = range(380, 781, 10)
    # Filter wavelengths to only include those present in spectral_data
    available_wvl = [w for w in photopic_wvl if w in spectral_data]
    if not available_wvl:
        raise ValueError("No spectral data available in the photopic range (380-780nm)")

    tvf = [spectral_data[w][0] for w in available_wvl]
    rvf = [spectral_data[w][1] for w in available_wvl]
    rvb = [spectral_data[w][2] for w in available_wvl]
    # Use the actual wavelength range from available data
    min_wvl = min(available_wvl)
    max_wvl = max(available_wvl)
    tf_x, tf_y, tf_z = pr.spec_xyz(tvf, min_wvl, max_wvl)
    rf_x, rf_y, rf_z = pr.spec_xyz(rvf, min_wvl, max_wvl)
    rb_x, rb_y, rb_z = pr.spec_xyz(rvb, min_wvl, max_wvl)
    tf_rgb = pr.xyz_rgb(tf_x, tf_y, tf_z)
    rf_rgb = pr.xyz_rgb(rf_x, rf_y, rf_z)
    rb_rgb = pr.xyz_rgb(rb_x, rb_y, rb_z)
    if coated_side == "front":
        coated_rgb = rf_rgb
        glass_rgb = rb_rgb
    else:
        coated_rgb = rb_rgb
        glass_rgb = rf_rgb
    return PaneRGB(coated_rgb, glass_rgb, tf_rgb, coated_side)


def get_layer_rgb(layer: pwc.ProductData) -> PaneRGB:
    """Get the RGB values for a pane layer."""
    photopic_wvl = range(380, 781, 10)
    if isinstance(layer.measurements, pwc.DualBandBSDF):
        raise ValueError("Dual band measurements not supported.")
    hemi = {
        int(round(d.wavelength * NM_PER_MM)): (
            d.direct_component.transmittance_front,
            d.direct_component.transmittance_back,
            d.direct_component.reflectance_front,
            d.direct_component.reflectance_back,
        )
        for d in layer.measurements
    }
    # Filter wavelengths to only include those present in hemi data
    available_wvl = [w for w in photopic_wvl if w in hemi]
    if not available_wvl:
        raise ValueError("No spectral data available in the photopic range (380-780nm)")

    tvf = [hemi[w][0] for w in available_wvl]
    rvf = [hemi[w][2] for w in available_wvl]
    rvb = [hemi[w][3] for w in available_wvl]
    # Use the actual wavelength range from available data
    min_wvl = min(available_wvl)
    max_wvl = max(available_wvl)
    tf_x, tf_y, tf_z = pr.spec_xyz(tvf, min_wvl, max_wvl)
    rf_x, rf_y, rf_z = pr.spec_xyz(rvf, min_wvl, max_wvl)
    rb_x, rb_y, rb_z = pr.spec_xyz(rvb, min_wvl, max_wvl)
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


def get_layer_groups(layers: Sequence[Layer]) -> list:
    if not layers or not isinstance(layers[0], Layer):
        raise ValueError()

    # Group consecutive glazing layers
    grouped_system: list[tuple[str, int]] = []
    current_group = 0
    current_type = ""

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
            current_type = ""

    # Add the last group if it exists
    if current_group:
        grouped_system.append((current_type, current_group))

    return grouped_system


def get_proxy_geometry(window: Polygon, gs: GlazingSystem) -> list[bytes]:
    FEPS = 1e-5
    layer_groups = get_layer_groups(gs.layers)
    window_vertices = window.vertices
    primitives: list[bytes] = []
    current_index = 0
    for group in layer_groups:
        if group[0] == "glazing":
            glazing_layer_count = group[1]
            # Get BRTDFunc
            mat_name = f"mat_{gs.name}_glazing_{current_index}"
            geom_name = f"{gs.name}_glazing_{current_index}"
            rgb = [
                get_pane_rgb(layer.spectral_data, layer.coated_side)
                for layer in gs.layers[
                    current_index : current_index + glazing_layer_count
                ]
            ]
            mat: pr.Primitive = get_glazing_primitive(mat_name, rgb)
            geom = polygon_primitive(window, mat_name, geom_name)
            primitives.append(mat.bytes)
            primitives.append(geom.bytes)
            current_index += glazing_layer_count
        elif group[0] == FABRIC:
            mat_name = f"mat_{gs.name}_fabric_{current_index}"
            geom_name = f"{gs.name}_fabric_{current_index}"
            xml_name = f"{gs.name}_fabric_{current_index}.xml"
            breakpoint()
            # Get aBSDF primitive
            with open(xml_name, "w") as f:
                f.write(gs.layers[current_index].shading_xml)
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
            width: float = 0
            height: float = 0
            if zdiff1 <= FEPS:
                width = float(dim1)
                height = float(dim2)
            elif zdiff2 <= FEPS:
                width = float(dim2)
                height = float(dim1)
            else:
                print("Error: Skewed window not supported: ")
                print(window_vertices)
            blinds_layer = gs.layers[current_index]
            geom = pr.BlindsGeometry(
                depth=blinds_layer.slat_width_m,
                width=width,
                height=height,
                nslats=blinds_layer.nslats,
                angle=blinds_layer.slat_angle_deg,
                rcurv=blinds_layer.slat_curve_m,
            )
            blinds: bytes = pr.generate_blinds(blinds_layer.shading_material, geom)
            xmin, xmax, ymin, ymax, zmin, zmax = pr.getbbox(blinds)
            blinds_normal = np.array((1, 0, 0))
            rotatez_angle = angle_between(blinds_normal, window.normal, degree=True)
            rotated_blinds: bytes = pr.Xform(blinds).rotatez(rotatez_angle)()
            xmin, xmax, ymin, ymax, zmin, zmax = pr.getbbox(rotated_blinds)
            rotated_blinds_centroid = np.array(
                ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
            )
            blinds_at_window = pr.Xform(rotated_blinds).translate(
                *(window.centroid - rotated_blinds_centroid)
            )()
            primitives.append(
                pr.Xform(blinds_at_window).translate(
                    *(window.normal * gs.layers[current_index].thickness_m)
                )()
            )
            current_index += 1
    return primitives


def get_spectral_multi_layer_optics(
    glazings: Sequence[Layer], name: str = "unnamed"
) -> bytes:
    """Generate spectral multi-layer optics data from glazing layers."""
    # Convert Layer objects to GlazingLayerData objects
    glazing_layers = []
    for layer in glazings:
        # Extract spectral points from layer spectral_data
        spectral_points = []
        for wavelength_nm, (t, rf, rb) in layer.spectral_data.items():
            spectral_points.append(
                pr.SpectralPoint(wavelength_nm=wavelength_nm, rf=rf, rb=rb, t=t)
            )

        # Sort by wavelength
        spectral_points.sort(key=lambda x: x.wavelength_nm)

        # Determine glazing type based on layer properties
        glazing_type = pr.GlazingType.monolithic
        if layer.coated_side:
            glazing_type = pr.GlazingType.coated

        glazing_layer = pr.GlazingLayerData(
            name=layer.product_name,
            glazing_type=glazing_type,
            thickness_m=layer.thickness_m,
            spectral_points=spectral_points,
        )
        glazing_layers.append(glazing_layer)

    # this call also generates .dat files in the current working directory
    return pr.genglaze_data(glazing_layers, prefix=name)


def generate_coplanar_bsdf(
    layers: list[Layer],
    gaps: list[Gap],
    outspec: pr.genbsdf.OutSpec = "y",
    nproc: int = 1,
    nspec: int = 3,
    nsamp: int = 2000,
):
    """Generate coplanar BSDF for the given layers and gaps."""
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            tmpdir_path = Path(tmpdir)
            layer_groups = get_layer_groups(layers)
            primitives = []
            width = 1
            height = 1
            total_thickness = sum(gap.thickness_m for gap in gaps)
            total_thickness += sum(layer.thickness_m for layer in layers)

            # Start from z=0 (innermost) and work outward (negative z)
            current_z = 0
            current_layer_idx = len(layers) - 1  # Start from last layer

            # Check if we have blinds for sampling box calculation
            has_blinds = any(group[0] == "blinds" for group in layer_groups)
            blinds_spacing = 0.01  # default

            for group_idx, group in enumerate(reversed(layer_groups)):
                # Create polygon at current z position
                polygon = Polygon(
                    [
                        np.array([0, 0, current_z]),
                        np.array([0, height, current_z]),
                        np.array([width, height, current_z]),
                        np.array([width, 0, current_z]),
                    ]
                )

                if group[0] == "glazing":
                    glazing_layer_count = group[1]
                    name = f"glazing_{group_idx}"

                    # Generate spectral data for glazing layers
                    glazing_layers = layers[
                        current_layer_idx - glazing_layer_count + 1 : current_layer_idx
                        + 1
                    ]
                    glazing_material_bytes = get_spectral_multi_layer_optics(
                        glazing_layers, name=name
                    )
                    glazing_material_primitives = parse_primitive(
                        glazing_material_bytes
                    )
                    glazing_material_name = glazing_material_primitives[-1].identifier

                    # Write spectral data to file in tmpdir
                    mat_file = tmpdir_path / f"{name}.dat"
                    with open(mat_file, "wb") as f:
                        f.write(glazing_material_bytes)

                    # Create geometry primitive
                    geom = polygon_primitive(polygon, glazing_material_name, name)
                    primitives.append(glazing_material_bytes)
                    primitives.append(b"")
                    primitives.append(geom.bytes)

                    current_layer_idx -= glazing_layer_count

                    # Move outward by gap thickness
                    gap_idx = (
                        len(layer_groups) - 1 - group_idx - 1
                    )  # Corresponding gap index
                    if gap_idx >= 0 and gap_idx < len(gaps):
                        current_z -= gaps[gap_idx].thickness_m

                elif group[0] == FABRIC:
                    fabric_layer = layers[current_layer_idx]

                    mat_name = f"mat_fabric_{group_idx}"
                    geom_name = f"fabric_{group_idx}"
                    xml_name = tmpdir_path / f"fabric_{group_idx}.xml"

                    # Write fabric XML to file in tmpdir
                    with open(xml_name, "w") as f:
                        f.write(fabric_layer.shading_xml)

                    # Create aBSDF primitive
                    mat = pr.Primitive(
                        "void",
                        "aBSDF",
                        mat_name,
                        [str(xml_name), "0", "0", "1", "."],
                        [],
                    )
                    geom = polygon_primitive(polygon, mat_name, geom_name)
                    primitives.extend([mat.bytes, geom.bytes])

                    current_layer_idx -= 1

                    # Move outward by gap thickness
                    gap_idx = (
                        len(layer_groups) - 1 - group_idx - 1
                    )  # Corresponding gap index
                    if gap_idx >= 0 and gap_idx < len(gaps):
                        current_z -= gaps[gap_idx].thickness_m

                elif group[0] == "blinds":
                    blinds_layer = layers[current_layer_idx]
                    blinds_spacing = blinds_layer.slat_spacing_m

                    geom_spec = pr.BlindsGeometry(
                        depth=blinds_layer.slat_width_m,
                        width=width,
                        height=height,
                        nslats=blinds_layer.nslats,
                        angle=blinds_layer.slat_angle_deg,
                        rcurv=blinds_layer.slat_curve_m,
                    )

                    # Generate blinds geometry
                    # 0 1 0 1 -thickness 0
                    blinds = pr.generate_blinds_for_bsdf(
                        blinds_layer.shading_material, geom_spec
                    )

                    # Position blinds at polygon location
                    xmin, xmax, ymin, ymax, zmin, zmax = pr.ot.getbbox(blinds)
                    thickness = zmax - zmin
                    blinds_centroid = np.array(
                        ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
                    )
                    translate_vector = polygon.centroid - blinds_centroid
                    translate_vector_x = translate_vector[0]
                    translate_vector_y = translate_vector[1]
                    translate_vector_z = translate_vector[2] - thickness / 2
                    positioned_blinds = pr.Xform(blinds).translate(
                        translate_vector_x,
                        translate_vector_y,
                        translate_vector_z,
                    )()
                    primitives.append(positioned_blinds)

                    current_layer_idx -= 1

                    # Move outward by gap thickness minus half blinds depth (for blinds space)
                    gap_idx = (
                        len(layer_groups) - 1 - group_idx - 1
                    )  # Corresponding gap index
                    if gap_idx >= 0 and gap_idx < len(gaps):
                        gap_thickness = gaps[gap_idx].thickness_m
                        current_z -= gap_thickness
                        current_z -= thickness

            device_file = tmpdir_path / "device.rad"
            with open(device_file, "wb") as f:
                for primitive in primitives:
                    f.write(primitive)

            box_height = blinds_spacing if has_blinds else 0.01
            sampling_box = pr.genbsdf.SamplingBox(
                xmin=0.5,
                xmax=0.51,
                ymin=0.5,
                ymax=0.5 + box_height,
                zmin=-total_thickness,
                zmax=0,
            )

            bsdf_result = pr.generate_bsdf(
                str(device_file),
                basis="kf",
                outspec=outspec,
                dim=sampling_box,
                nproc=nproc,
                nspec=nspec,
                nsamp=nsamp,
            )
        finally:
            os.chdir(original_dir)
    return bsdf_result


def generate_melanopic_bsdf(
    layers: list[Layer],
    gaps: list[Gap],
    nproc: int = 1,
    nsamp: int = 2000,
    nspec: int = 20,
) -> tuple[list[list], list[list]]:
    """Generate melanopic BSDF for the glazing system."""
    data = generate_coplanar_bsdf(
        layers, gaps, outspec="m", nspec=nspec, nproc=nproc, nsamp=nsamp
    )
    back_transmittance = [
        list(map(float, row.split()))
        for row in data.back.transmittance.decode().splitlines()
    ]
    back_reflectance = [
        list(map(float, row.split()))
        for row in data.back.reflectance.decode().splitlines()
    ]
    return back_transmittance, back_reflectance
