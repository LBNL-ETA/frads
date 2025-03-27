import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import math

import pyradiance as pr
import pywincalc as pwc

AIR = pwc.PredefinedGasType.AIR
KRYPTON = pwc.PredefinedGasType.KRYPTON
XENON = pwc.PredefinedGasType.XENON
ARGON = pwc.PredefinedGasType.ARGON


@dataclass
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


@dataclass
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
    thickness: float
    product_type: str
    conductivity: float
    emissivity_front: float
    emissivity_back: float
    ir_transmittance: float
    top_opening_multiplier: float = 0
    bottom_opening_multiplier: float = 0
    left_side_opening_multiplier: float = 0
    right_side_opening_multiplier: float = 0
    front_opening_multiplier: float = 0.05
    slat_width: float = 0.0160
    slat_spacing: float = 0.0120
    slat_thickness: float = 0.0
    slat_angle: float = 90.0
    slat_conductivity: float = 160.00
    slat_curve: float = 0.0
    flipped: bool = False
    rgb: PaneRGB


@dataclass
class LayerInput:
    input: Path | str | bytes
    top_opening_multiplier: float = 0
    bottom_opening_multiplier: float = 0
    left_side_opening_multiplier: float = 0
    right_side_opening_multiplier: float = 0
    front_opening_multiplier: float = 0.05
    slat_angle: float = 0
    flipped: bool = False


@dataclass
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


@dataclass
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


@dataclass
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
    # blinds_index: int = -1
    # fabric_index: int = -1

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
            _vi = pr.WrapBSDFInput("Visible", _tvf, _tvb, _rvf, _rvb)
            _si = pr.WrapBSDFInput("Solar", _tsf, _tsb, _rsf, _rsb)
            with open(out, "wb") as f:
                f.write(
                    pr.wrapbsdf(
                        enforce_window=True,
                        basis="kf",
                        inp=[_vi, _si],
                        n=self.name,
                        m="",
                        t=str(self.thickness),
                    )
                )

    def save(self, out: Union[str, Path]):
        """Save the glazing system to a file."""
        out = Path(out)
        if out.suffix == ".xml":
            self.to_xml(out)
        else:
            with open(out.with_suffix(".json"), "w") as f:
                json.dump(asdict(self), f)

    @classmethod
    def from_json(cls, path: Union[str, Path]):
        """Load a glazing system from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        layers = data.pop("layers")
        layer_instances = []
        for layer in layers:
            rgb = layer.pop("rgb")
            layer_instances.append(Layer(rgb=PaneRGB(**rgb), **layer))
        gaps = data.pop("gaps")
        gap_instances = []
        for gap in gaps:
            gas_instances = []
            gases = gap.pop("gas")
            for gs in gases:
                gas_instances.append(Gas(**gs))
            gap_instances.append(Gap(gas=gas_instances, **gap))
        return cls(layers=layer_instances, gaps=gap_instances, **data)

    def get_brtdfunc(self) -> pr.Primitive:
        """Get a BRTDfunc primitive for the glazing system."""
        if any(layer.product_type != "glazing" for layer in self.layers):
            raise ValueError("Only glass layers supported.")
        if len(self.layers) > 2:
            raise ValueError("Only double pane supported.")
        rgb = [layer.rgb for layer in self.layers]
        return get_glazing_primitive(self.name, rgb)


def load_glazing_system(path: Union[str, Path]):
    """Load a glazing system from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    layers = data.pop("layers")
    layer_instances = []
    for layer in layers:
        rgb = layer.pop("rgb")
        layer_instances.append(Layer(rgb=PaneRGB(**rgb), **layer))
    gaps = data.pop("gaps")
    gap_instances = []
    for gap in gaps:
        gas_instances = []
        gases = gap.pop("gas")
        for gs in gases:
            gas_instances.append(Gas(**gs))
        gap_instances.append(Gap(gas=gas_instances, **gap))
    return GlazingSystem(layers=layer_instances, gaps=gap_instances, **data)


def get_layer_data(inp: pwc.ProductData, layer_inp: LayerInput) -> Layer:
    """Create a list of layers from a list of pwc.ProductData."""
    return Layer(
        product_name=inp.product_name,
        thickness=inp.thickness,
        product_type=inp.product_type,
        conductivity=inp.conductivity,
        emissivity_front=inp.emissivity_front,
        emissivity_back=inp.emissivity_back,
        ir_transmittance=inp.ir_transmittance,
        rgb=get_layer_rgb(inp),
        top_opening_multiplier=layer_inp.top_opening_multiplier,
        bottom_opening_multiplier=layer_inp.bottom_opening_multiplier,
        left_side_opening_multiplier=layer_inp.left_side_opening_multiplier,
        right_side_opening_multiplier=layer_inp.right_side_opening_multiplier,
        front_opening_multiplier=layer_inp.front_opening_multiplier,
        slat_angle=layer_inp.slat_angle,
        flipped=layer_inp.flipped,
    )


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


def generate_blinds_xml(
    depth,
    nslats,
    angle,
    curvature,
    thickness,
    name,
    r_sol_diff,
    r_sol_spec,
    r_vis_diff,
    r_vis_spec,
    r_ir,
    nproc,
):
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
    sol_blinds = pr.generate_blinds(material_solar, geom)
    vis_blinds = pr.generate_blinds(material_visible, geom)
    ir_blinds = pr.generate_blinds(material_ir, geom)
    sol_results = pr.generate_bsdf(sol_blinds, nproc=nproc)
    vis_results = pr.generate_bsdf(vis_blinds, nproc=nproc)
    ir_results = pr.generate_bsdf(ir_blinds, basis="u")
    return pr.generate_xml(
        sol_results, vis_results, ir_results, n=name, m="unnamed", t=thickness
    )


def get_dual_band_values_from_blinds_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    dual_band_values = data["composition"][0]["child_product"]["spectral_data"][
        "dual_band_values"
    ]
    rf_sol_diff = dual_band_values["Rf_sol_diffuse"]
    rf_sol_spec = dual_band_values["Rf_sol_specular"]
    rf_vis_diff = dual_band_values["Rf_vis_diffuse"]
    rf_vis_spec = dual_band_values["Rf_vis_specular"]
    return rf_sol_diff, rf_sol_spec, rf_vis_diff, rf_vis_spec


def create_glazing_system(
    name: str,
    layer_inputs: list[LayerInput],
    gaps: None | list[Gap] = None,
    nproc: int = 1,
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
    # blinds_index = -1
    # fabric_index = -1
    if gaps is None:
        gaps = [Gap([Gas("air", 1)], 0.0127) for _ in range(len(layer_inputs) - 1)]
    layer_data: list[Layer] = []
    thickness = 0
    for idx, layer_inp in enumerate(layer_inputs):
        product_data = None
        layer = Layer()
        if isinstance(layer_inp.input, str) or isinstance(layer_inp.input, Path):
            if not os.path.isfile(layer_inp.input):
                raise FileNotFoundError(f"{layer_inp} does not exist.")
            if isinstance(layer_inp.input, Path):
                layer_inp.input = str(layer_inp.input)
            if layer_inp.input.endswith(".json"):
                product_data = pwc.parse_json_file(layer_inp.input)
            elif layer_inp.input.endswith(".xml"):
                product_data = pwc.parse_bsdf_xml_file(layer_inp.input)
                if product_data.product_type is None:
                    product_data.product_type = "shading"
            else:
                product_data = pwc.parse_optics_file(layer_inp.input)
        elif isinstance(layer_inp.input, bytes):
            try:
                product_data = pwc.parse_json(layer_inp.input)
            except json.JSONDecodeError:
                product_data = pwc.parse_bsdf_xml_string(layer_inp.input)
                if product_data.product_type is None:
                    product_data.product_type = "shading"
        if product_data is None:
            raise ValueError("Invalid layer type")
        if product_data.product_type == "shading":
            if product_data.product_subtype == "venetian":
                rf_sol_diff, rf_sol_spec, rf_vis_diff, rf_vis_spec = (
                    get_dual_band_values_from_blinds_json(layer_inp.input)
                )
                slat_spacing = product_data.composition.geometry.slat_spacing / 1e3
                tir = product_data.composition.material.ir_transmittance
                nslats = int(1 / slat_spacing)
                slat_depth = (product_data.composition.geometry.slat_width / 1e3,)
                slat_curvature = (
                    product_data.composition.geometry.slat_curvature / 1e3,
                )
                emis_front = product_data.composition.material.emissivity_front
                emis_back = product_data.composition.material.emissivity_back
                name = product_data.product_name
                layer_thickness = slat_depth * math.cos(
                    math.radians(layer_inp.slat_angle)
                )
                if tir != 0:
                    print("tir not zero")
                    return
                if emis_front != emis_back:
                    print("front and back emissivity not the same")
                    return
                rf_ir = 1 - tir - emis_front

                xml = generate_blinds_xml(
                    slat_depth,
                    nslats,
                    layer_inp.slat_angle,
                    slat_curvature,
                    layer_thickness,
                    name,
                    rf_sol_diff,
                    rf_sol_spec,
                    rf_vis_diff,
                    rf_vis_spec,
                    rf_ir,
                    nproc,
                )
                # blinds_index = idx
                real_product_data = pwc.parse_bsdf_xml_string(xml)
                layer = get_layer_data(real_product_data, layer_inp)
                layer_data.append(layer)
                thickness += layer_thickness
            else:
                # fabric_index = idx
                layer = get_layer_data(product_data, layer_inp)
                layer_data.append(layer)
                product_data.thickness = (
                    product_data.thickness / 1000.0 or 0.001
                )  # mm to m
                thickness += layer.thickness
        else:
            layer = get_layer_data(product_data, layer_inp)
            layer_data.append(layer)
            thickness += product_data.thickness

    for gap in gaps:
        thickness += gap.thickness

    glzsys = pwc.GlazingSystem(
        solid_layers=layer_data,
        gap_layers=create_pwc_gaps(gaps),
        width_meters=1,
        height_meters=1,
        environment=pwc.nfrc_shgc_environments(),
        bsdf_hemisphere=pwc.BSDFHemisphere.create(pwc.BSDFBasisType.FULL),
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
        # blinds_index=blinds_index,
        # fabric_index=fabric_index,
    )


def get_layer_rgb(layer: pwc.ProductData) -> PaneRGB:
    """Get the RGB values for a pane layer."""
    photopic_wvl = range(380, 781, 10)
    if isinstance(layer.measurements, pwc.DualBandBSDF):
        return PaneRGB(
            coated_rgb=(0, 0, 0),
            glass_rgb=(0, 0, 0),
            trans_rgb=(0, 0, 0),
            coated_side=None,
        )
    hemi = {
        int(round(d.wavelength * 1e3)): (
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
    if len(panes) > 2:
        raise ValueError("Only double pane supported")
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
    else:
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
    return pr.Primitive("void", "BRTDfunc", name, str_arg, real_arg)
