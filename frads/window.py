from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple, Union

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

    coated_rgb: Tuple[float, float, float]
    glass_rgb: Tuple[float, float, float]
    trans_rgb: Tuple[float, float, float]
    coated_side: Optional[str] = None


@dataclass
class MeasurementComponent:
    transmittance_front: float
    transmittance_back: float
    reflectance_front: float
    reflectance_back: float


@dataclass
class WavelengthData:
    direct_component: Optional[MeasurementComponent]
    diffuse_component: Optional[MeasurementComponent]


@dataclass
class Layer:
    product_name: str
    thickness: float
    product_type: str
    conductivity: float
    emissivity_front: float
    emissivity_back: float
    ir_transmittance: float


@dataclass
class GlazingLayer(Layer):
    measurements: Dict[float, WavelengthData]
    rgb: PaneRGB


@dataclass
class ShadingLayer(Layer):
    measurements: Dict[float, WavelengthData]


@dataclass
class Gas:
    gas: str
    ratio: float

    def __post_init__(self):
        if self.ratio < 0 or self.ratio > 1:
            raise ValueError("Gas ratio must be between 0 and 1.")
        if self.gas.lower() not in ("air", "argon", "krypton", "xenon"):
            raise ValueError("Invalid gas type.")


@dataclass
class Gap:
    gas: List[Gas]
    thickness: float

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError("Gap thickness must be greater than 0.")
        if sum(g.ratio for g in self.gas) != 1:
            raise ValueError("The sum of the gas ratios must be 1.")


@dataclass
class GlazingSystem:
    name: str
    thickness: float = 0
    layers: List[str] = field(default_factory=list)
    shading_layers: Dict[str, ShadingLayer] = field(default_factory=dict)
    glazing_layers: Dict[str, GlazingLayer] = field(default_factory=dict)
    gaps: List[Gap] = field(default_factory=list)
    visible_front_transmittance: List[List[float]] = field(default_factory=list)
    visible_back_transmittance: List[List[float]] = field(default_factory=list)
    visible_front_reflectance: List[List[float]] = field(default_factory=list)
    visible_back_reflectance: List[List[float]] = field(default_factory=list)
    solar_front_transmittance: List[List[float]] = field(default_factory=list)
    solar_back_transmittance: List[List[float]] = field(default_factory=list)
    solar_front_reflectance: List[List[float]] = field(default_factory=list)
    solar_back_reflectance: List[List[float]] = field(default_factory=list)
    solar_front_absorptance: List[List[float]] = field(default_factory=list)
    solar_back_absorptance: List[List[float]] = field(default_factory=list)

    def _matrix_to_str(self, matrix: List[List[float]]) -> str:
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

    def save(self, out: Union[str, Path]) -> None:
        """Save the glazing system to a file.
        If the file extension is .xml, save as an XML file.
        Otherwise, save as a JSON file.

        Args:
            out: The path to save the file to.
        """
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
        shading_layers = data.pop("shading_layers")
        glazing_layers = data.pop("glazing_layers")
        shading_layer_instances = {}
        glazing_layer_instances = {}
        for layer in glazing_layers:
            rgb = layer.pop("rgb")
            measurements = layer.pop("measurements")
            glazing_layer_instances[layer.name] = GlazingLayer(
                rgb=PaneRGB(**rgb), **layer
            )
        for layer in shading_layers:
            measurements = layer.pop("measurements")
            shading_layer_instances[layer.name] = ShadingLayer(**layer)
        gaps = data.pop("gaps")
        gap_instances = []
        for gap in gaps:
            gas_instances = []
            gases = gap.pop("gas")
            for gs in gases:
                gas_instances.append(Gas(**gs))
            gap_instances.append(Gap(gas=gas_instances, **gap))
        return cls(
            shading_layers=shading_layer_instances,
            glazing_layers=glazing_layer_instances,
            gaps=gap_instances,
            **data,
        )

    def get_brtdfunc(self) -> pr.Primitive:
        """Get a BRTDfunc primitive for the glazing system."""
        if not all(isinstance(layer, GlazingLayer) for layer in self.layers):
            raise ValueError("Only glazing layers supported.")
        if len(self.layers) > 2:
            raise ValueError("Only double pane supported.")
        rgb = [self.glazing_layers[name].rgb for name in self.layers]
        return get_glazing_primitive(self.name, rgb)


def get_glazing_measurements(
    measurements: List[pwc.WavelengthData],
) -> Dict[float, WavelengthData]:
    """Get the measurements from a pwc.ProductMeasurements object."""
    parsed = {}
    for data in measurements:
        _dc = None
        _df = None
        if (ddc := data.direct_component) is not None:
            _dc = MeasurementComponent(
                ddc.transmittance_front,
                ddc.transmittance_back,
                ddc.reflectance_front,
                ddc.reflectance_back,
            )
        if (dfc := data.diffuse_component) is not None:
            _df = MeasurementComponent(
                dfc.transmittance_front,
                dfc.transmittance_back,
                dfc.reflectance_front,
                dfc.reflectance_back,
            )
        parsed[data.wavelength * 1e3] = WavelengthData(
            direct_component=_dc, diffuse_component=_df
        )
    return parsed


def get_shading_measurements(
    measurements,
) -> Dict[float, WavelengthData]:
    """Get the measurements from a pwc.ProductMeasurements object."""
    return {
        d.wavelength
        * 1e3: WavelengthData(
            direct_component=d.direct_component.transmittance_front,
            diffuse_component=d.diffuse_component.transmittance_front,
        )
        for d in measurements
    }


def get_layers(input: List[pwc.ProductData]) -> List[Layer]:
    """Create a list of layers from a list of pwc.ProductData."""
    layers = []
    for inp in input:
        params = {
            "product_name": inp.product_name,
            "thickness": inp.thickness,
            "product_type": inp.product_type,
            "conductivity": inp.conductivity,
            "emissivity_front": inp.emissivity_front,
            "emissivity_back": inp.emissivity_back,
            "ir_transmittance": inp.ir_transmittance,
        }
        if inp.product_type.lower() == "glazing":
            layers.append(
                GlazingLayer(
                    rgb=get_layer_rgb(inp),
                    measurements=get_glazing_measurements(inp.measurements),
                    **params,
                )
            )
        else:
            layers.append(
                ShadingLayer(
                    measurements=get_shading_measurements(inp.measurements),
                    **params,
                )
            )
    return layers


def create_pwc_gaps(gaps: List[Gap]) -> List[pwc.Layers.gap]:
    """Create a list of pwc gaps from a list of gaps."""
    pwc_gaps = []
    for gap in gaps:
        _gas = pwc.create_gas(
            [[g.ratio, getattr(pwc.PredefinedGasType, g.gas.upper())] for g in gap.gas]
        )
        _gap = pwc.Layers.gap(gas=_gas, thickness=gap.thickness)
        pwc_gaps.append(_gap)
    return pwc_gaps


def get_default_gaps(nlayers: int) -> List[Gap]:
    """Get a list of default gaps."""
    return [Gap([Gas("air", 1)], 0.0127) for _ in range(nlayers - 1)]


def get_solar_photopic_results(
    layers: List[pwc.ProductData], gaps: List[pwc.Layers.gap]
) -> Dict[str, List[List[float]]]:
    """Get the solar and photopic results.

    Args:
        layers: A list of pwc.ProductData objects.
        gaps: A list of pwc.Layers.gap objects.

    Returns:
        A tuple of pwc.OpticalMethodResults objects.
    """
    glzsys = pwc.GlazingSystem(
        solid_layers=layers,
        gap_layers=gaps,
        width_meters=1,
        height_meters=1,
        environment=pwc.nfrc_shgc_environments(),
        bsdf_hemisphere=pwc.BSDFHemisphere.create(pwc.BSDFBasisType.FULL),
    )
    solres = glzsys.optical_method_results("SOLAR")
    visres = glzsys.optical_method_results("PHOTOPIC")
    solsys = solres.system_results
    vissys = visres.system_results

    return {
        "solar_front_absorptance": [
            alpha.front.absorptance.angular_total for alpha in solres.layer_results
        ],
        "solar_back_absorptance": [
            alpha.back.absorptance.angular_total for alpha in solres.layer_results
        ],
        "visible_back_reflectance": vissys.back.reflectance.matrix,
        "visible_front_reflectance": vissys.front.reflectance.matrix,
        "visible_back_transmittance": vissys.back.transmittance.matrix,
        "visible_front_transmittance": vissys.front.transmittance.matrix,
        "solar_back_reflectance": solsys.back.reflectance.matrix,
        "solar_front_reflectance": solsys.front.reflectance.matrix,
        "solar_back_transmittance": solsys.back.transmittance.matrix,
        "solar_front_transmittance": solsys.front.transmittance.matrix,
    }


def create_glazing_system(
    name: str,
    layers: List[Union[GlazingLayer, ShadingLayer]],
    gaps: Optional[List[Gap]] = None,
) -> GlazingSystem:
    """Create a glazing system from a list of layers and gaps.

    Args:
        name: The name of the glazing system.
        layers: A list of GlazingLayer or ShadingLayer objects.
        gaps: A list of Gap objects.

    Returns:
        A GlazingSystem object.
    """
    if gaps is None:
        gaps = get_default_gaps(len(layers))

    thickness = sum(layer.thickness for layer in layers)
    shading_layers = {}
    glazing_layers = {}
    for layer in layers:
        if isinstance(layer, ShadingLayer):
            shading_layers[layer.product_name] = layer
        elif isinstance(layer, GlazingLayer):
            glazing_layers[layer.product_name] = layer
        else:
            raise ValueError("Invalid layer type")
    layer_by_names = [layer.product_name for layer in layers]
    results = get_solar_photopic_results(
        layers=[layer.to_pwc() for layer in layers],
        gaps=create_pwc_gaps(gaps),
    )

    return GlazingSystem(
        name=name,
        thickness=thickness,
        layers=layer_by_names,
        shading_layers=shading_layers,
        glazing_layers=glazing_layers,
        gaps=gaps,
        **results,
    )


def create_glazing_system_from_file(
    name: str, layers: List[Union[Path, bytes]], gaps: Optional[List[Gap]] = None
) -> GlazingSystem:
    """Create a glazing system from a list of layers as fils and gaps.

    Args:
        name: The name of the glazing system.
        layers: A list of Path or bytes objects.
        gaps: A list of Gap objects.

    Returns:
        A GlazingSystem object.
    """
    if gaps is None:
        gaps = [Gap([Gas("air", 1)], 0.0127) for _ in range(len(layers) - 1)]
    layer_data = []
    thickness = 0
    for layer in layers:
        product_data = None
        if isinstance(layer, Path):
            if layer.suffix == ".json":
                product_data = pwc.parse_json_file(str(layer))
            elif layer.suffix == ".xml":
                product_data = pwc.parse_bsdf_xml_file(str(layer))
            else:
                product_data = pwc.parse_optics_file(str(layer))
        elif isinstance(layer, bytes):
            try:
                product_data = pwc.parse_json(layer)
            except json.JSONDecodeError:
                product_data = pwc.parse_bsdf_xml_string(layer)
        if product_data is None:
            raise ValueError("Invalid layer type")
        layer_data.append(product_data)
        thickness += product_data.thickness / 1000.0 or 0  # mm to m

    pwc_gaps = create_pwc_gaps(gaps)
    results = get_solar_photopic_results(layer_data, pwc_gaps)

    return GlazingSystem(
        name=name,
        thickness=thickness,
        layers=get_layers(layer_data),
        shading_layers={},
        glazing_layers={},
        gaps=gaps,
        **results,
    )


def get_pane_rgb(layer: pwc.ProductData) -> PaneRGB:
    """Get the RGB values for a pane layer."""
    photopic_wvl = range(380, 781, 10)
    hemi = {
        d.wavelength
        * 1e3: (
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


def get_glazing_primitive(name: str, panes: List[PaneRGB]) -> pr.Primitive:
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
    else:
        raise ValueError("Only double pane supported")
    return pr.Primitive("void", "BRTDfunc", name, str_arg, real_arg)


def laminate(sub: GlazingLayer, lam: GlazingLayer, side: pwc.CoatedSide) -> GlazingLayer:
    """Laminate a glazing layer."""
    optical_data = pwc.ProductDataOpticalNBand()
    layer = pwc.ProductDataOpticalAndThermal()
    return GlazingLayer(
        product_name=sub.product_name,
        thickness=sub.thickness,
        product_type=sub.product_type,
        conductivity=sub.conductivity,
        emissivity_front=sub.emissivity_front,
        emissivity_back=sub.emissivity_back,
        ir_transmittance=sub.ir_transmittance,
        measurements=sub.measurements,
        rgb=sub.rgb,
    )
