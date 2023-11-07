from enum import Enum
import json
from pathlib import Path
import re
import tempfile
from typing import Dict, List, Optional, Tuple, Union

from lxml import etree
import numpy as np
from pydantic import BaseModel, field_validator
import pyradiance as pr
import pywincalc as wc


class GasType(str, Enum):
    air = "air"
    krypton = "krypton"
    xenon = "xenon"
    argon = "argon"


class OpticalComponent(Enum):
    transmittance = "transmittance"
    reflectance = "reflectance"


class OpticalDirection(Enum):
    front = "front"
    back = "back"


class WavelengthIntegral(Enum):
    solar = "solar"
    visible = "visible"


class WavelengthData(BaseModel):
    component: OpticalComponent
    direction: OpticalDirection
    wavelength: WavelengthIntegral
    data: List[float]


class RGBFloat(BaseModel):
    red: float
    green: float
    blue: float


class ShadingXMLParser:
    def __init__(self, file_path):
        self.name = ""
        self.manufacturer = ""
        self.element_type = ""
        self.thickness = 0.0
        self.device_type = None
        self.thermal_conductivity = None
        self.emissivity_front = None
        self.emissivity_back = None
        self.tir = None
        self.permeability_factor = None
        self.aerc_acceptance = None
        self.angle_basis_name = ""
        self.thetas = []
        self.nphis = []
        self.wavelength_data = []
        self.current_direction = None
        self.current_component = None
        self.current_spectrum = None

        context = etree.iterparse(file_path, events=("end",))

        for event, element in context:
            tag = element.tag.split("}")[-1]

            if handler := getattr(self, "_handle_" + tag, None):
                handler(element)

            element.clear()

    def _handle_WindowElementType(self, element):
        self.element_type = element.text

    def _handle_Name(self, element):
        self.name = element.text.strip()

    def _handle_Manufacturer(self, element):
        self.manufacturer = element.text.strip()

    def _handle_Thickness(self, element):
        self.thickness = float(element.text)

    def _handle_DeviceType(self, element):
        self.device_type = element.text.strip()

    def _handle_ThermalConductivity(self, element):
        self.thermal_conductivity = float(element.text)

    def _handle_EmissivityFront(self, element):
        self.emissivity_front = float(element.text)

    def _handle_EmissivityBack(self, element):
        self.emissivity_back = float(element.text)

    def _handle_TIR(self, element):
        self.tir = float(element.text)

    def _handle_PermeabilityFactor(self, element):
        self.permeability_factor = float(element.text)

    def _handle_AERCAcceptance(self, element):
        self.aerc_acceptance = element.text

    def _handle_AngleBasisName(self, element):
        self.angle_basis_name = element.text

    def _handle_Theta(self, element):
        self.thetas.append(float(element.text))

    def _handle_nPhis(self, element):
        self.nphis.append(int(element.text))

    def _handle_WavelengthDataDirection(self, element):
        comp, direction = element.text.strip().split()
        self.current_direction = direction.lower()
        if comp.lower() == "transmission":
            self.current_component = "transmittance"
        elif comp.lower() == "reflection":
            self.current_component = "reflectance"
        else:
            raise ValueError(f"Unknown component: {comp}")

    def _handle_Wavelength(self, element):
        unit = element.attrib.get("unit")
        self.current_spectrum = element.text.strip().lower()

    def _handle_ScatteringData(self, element):
        scattering_data = [
            float(x) for x in re.split("[, \\n]", element.text.strip()) if x
        ]
        self.wavelength_data.append(
            WavelengthData(
                component=self.current_component,
                direction=self.current_direction,
                wavelength=self.current_spectrum,
                data=scattering_data,
            )
        )


class ShadingBSDF(BaseModel):
    name: str
    manufacturer: str
    element_type: str
    thickness: float
    device_type: Optional[str]
    thermal_conductivity: Optional[float]
    emissivity_front: Optional[float]
    emissivity_back: Optional[float]
    tir: Optional[float]
    permeability_factor: Optional[float]
    aerc_acceptance: Optional[str]
    angle_basis_name: str
    thetas: Optional[List[float]]
    nphis: Optional[List[int]]
    wavelength_data: List[WavelengthData]

    @classmethod
    def from_xml(cls, file_path):
        parser = ShadingXMLParser(file_path)
        return cls(
            name=parser.name,
            manufacturer=parser.manufacturer,
            element_type=parser.element_type,
            thickness=parser.thickness,
            device_type=parser.device_type,
            thermal_conductivity=parser.thermal_conductivity,
            emissivity_front=parser.emissivity_front,
            emissivity_back=parser.emissivity_back,
            tir=parser.tir,
            permeability_factor=parser.permeability_factor,
            aerc_acceptance=parser.aerc_acceptance,
            angle_basis_name=parser.angle_basis_name,
            thetas=parser.thetas,
            nphis=parser.nphis,
            wavelength_data=parser.wavelength_data,
        )

    def to_wincalc(self, flip=False) -> wc.ProductDataOpticalAndThermal:
        data = {}
        if self.nphis is None:
            raise ValueError("nphis not found, may be not a matrix")
        if len(self.wavelength_data) != 8:
            raise ValueError("Wavelength data incomplete")
        nrows = sum(self.nphis)
        for wd in self.wavelength_data:
            key = f"{wd.wavelength.value}_{wd.component.value}_{wd.direction.value}"
            data[key] = [wd.data[i : i + nrows] for i in range(0, len(wd.data), nrows)]
        optical_data = wc.ProductDataOpticalDualBandBSDF(
            bsdf_hemisphere=wc.BSDFHemisphere.create(wc.BSDFBasisType.FULL),
            thickness_meters=self.thickness / 1000,
            ir_transmittance_front=self.tir,
            ir_transmittance_back=self.tir,
            emissivity_front=self.emissivity_front,
            emissivity_back=self.emissivity_back,
            permeability_factor=self.permeability_factor,
            flipped=flip,
            **data,
        )
        thermal_data = wc.ProductDataThermal(
            conductivity=self.thermal_conductivity,
            thickness_meters=self.thickness / 1000,
            flipped=flip,
        )
        return wc.ProductDataOpticalAndThermal(optical_data, thermal_data)


class IntegratedResultsSummary(BaseModel):
    calculation_standard_name: str
    tfsol: float
    tbsol: Optional[float]
    rfsol: float
    rbsol: float
    tfvis: float
    tbvis: Optional[float]
    rfvis: float
    rbvis: float
    tdw: float
    tuv: float
    tspf: float
    tkr: Optional[float]
    tciex: float
    tciey: float
    tciez: float
    tf_r: Optional[float]
    tf_g: Optional[float]
    tf_b: Optional[float]
    rfciex: float
    rfciey: float
    rfciez: float
    rf_r: Optional[float]
    rf_g: Optional[float]
    rf_b: Optional[float]
    rbciex: Optional[float]
    rbciey: Optional[float]
    rbciez: Optional[float]
    rb_r: Optional[float]
    rb_g: Optional[float]
    rb_b: Optional[float]


class MeasuredData(BaseModel):
    is_specular: bool
    thickness: float
    tir_front: float
    tir_back: Optional[float]
    emissivity_front: float
    emissivity_back: float
    conductivity: float
    permeability_factor: Optional[float]


class SpectralData(BaseModel):
    T: float
    Rf: float
    Rb: float
    wavelength: float


class Pane(BaseModel):
    product_id: int
    name: str
    product_name: Optional[str]
    nfrc_id: int
    igdb_database_version: str
    acceptance: str
    appearance: str
    manufacturer_name: str
    thickness: float
    short_description: Optional[str]
    type: str
    subtype: str
    deconstructable: bool
    coating_name: str
    coated_side: str
    measured_data: MeasuredData
    integrated_results_summary: List[IntegratedResultsSummary]
    spectral_data: Dict[str, Union[List[SpectralData], dict]]
    composition: Optional[list]
    coated_side_rgb: RGBFloat
    glass_side_rgb: RGBFloat
    transmittance_rgb: RGBFloat
    substrate: Optional["Pane"]

    @classmethod
    def from_json(cls, file_path: Union[str, Path]):
        with open(file_path, "r") as f:
            data = json.load(f)
        sdata = [SpectralData(**d) for d in data["spectral_data"]["spectral_data"]]
        trgb, crgb, grgb = get_pane_rgb(sdata, data['coated_side'])
        data["transmittance_rgb"] = trgb
        data["coated_side_rgb"] = crgb
        data["glass_side_rgb"] = grgb
        data["thickness"] = data["measured_data"]["thickness"]
        return cls.model_validate(data)

    def to_wincalc(self, flip=False) -> wc.ProductDataOpticalAndThermal:
        optical_data = wc.ProductDataOpticalNBand(
            material_type=getattr(
                wc.MaterialType, self.subtype.upper().replace("-", "_")
            ),
            thickness_meters=self.measured_data.thickness / 1e3,
            wavelength_data=[
                wc.WavelengthData(
                    s.wavelength, wc.OpticalMeasurementComponent(s.T, s.T, s.Rf, s.Rb)
                )
                for s in self.spectral_data["spectral_data"]
            ],
            coated_side=getattr(wc.CoatedSide, self.coated_side.upper()),
            ir_transmittance_front=self.measured_data.tir_front,
            ir_transmittance_back=self.measured_data.tir_back
            or self.measured_data.tir_front,
            emissivity_front=self.measured_data.emissivity_front,
            emissivity_back=self.measured_data.emissivity_back,
            permeability_factor=self.measured_data.permeability_factor or 0,
            flipped=flip,
        )
        thermal_data = wc.ProductDataThermal(
            conductivity=self.measured_data.conductivity,
            thickness_meters=self.measured_data.thickness / 1e3,
            flipped=flip,
        )
        return wc.ProductDataOpticalAndThermal(optical_data, thermal_data)

    def to_array(self) -> np.ndarray:
        """Convert the pane to a 2D array."""
        return np.array(
            [
                [s.T, s.Rf, s.Rb]
                for s in self.spectral_data["spectral_data"]
            ]
        )

    def to_brtdfunc(self, name=None) -> pr.Primitive:
        real_arg = [0.0] * 9
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
        real_arg.extend([
            1 if self.coated_side.lower() == "front" else -1,
            self.glass_side_rgb.red,
            self.glass_side_rgb.green,
            self.glass_side_rgb.blue,
            self.coated_side_rgb.red,
            self.coated_side_rgb.green,
            self.coated_side_rgb.blue,
            self.transmittance_rgb.red,
            self.transmittance_rgb.green,
            self.transmittance_rgb.blue,
        ])
        return pr.Primitive("void", "BRTDfunc", name or self.name, str_arg, real_arg)


def get_pane_rgb(
    sdata: List[SpectralData],
    coated_side: str
) -> Tuple[RGBFloat, RGBFloat, RGBFloat]:
    """Get the RGB values for a pane layer."""
    photopic_wvl = range(380, 781, 10)
    hemi = {d.wavelength * 1e3: (d.T, d.T, d.Rf, d.Rb) for d in sdata}
    tvf = [hemi[w][0] for w in photopic_wvl]
    rvf = [hemi[w][2] for w in photopic_wvl]
    rvb = [hemi[w][3] for w in photopic_wvl]
    tf_x, tf_y, tf_z = pr.spec_xyz(tvf, 380, 780)
    rf_x, rf_y, rf_z = pr.spec_xyz(rvf, 380, 780)
    rb_x, rb_y, rb_z = pr.spec_xyz(rvb, 380, 780)
    tf_rgb = pr.xyz_rgb(tf_x, tf_y, tf_z)
    rf_rgb = pr.xyz_rgb(rf_x, rf_y, rf_z)
    rb_rgb = pr.xyz_rgb(rb_x, rb_y, rb_z)
    if coated_side.lower() == "front":
        coated_rgb = rf_rgb
        glass_rgb = rb_rgb
    else:
        coated_rgb = rb_rgb
        glass_rgb = rf_rgb
    trgb = RGBFloat(red=tf_rgb[0], green=tf_rgb[1], blue=tf_rgb[2])
    crgb = RGBFloat(red=coated_rgb[0], green=coated_rgb[1], blue=coated_rgb[2])
    grgb = RGBFloat(red=glass_rgb[0], green=glass_rgb[1], blue=glass_rgb[2])
    return trgb, crgb, grgb


class Gas(BaseModel):
    gas: GasType
    ratio: float

    @field_validator("ratio")
    @classmethod
    def check_ratio(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Gas ratio must be between 0 and 1.")


class Gap(BaseModel):
    gases: List[Gas]
    thickness: float

    @field_validator("gases")
    @classmethod
    def check_gases(cls, v):
        if sum(gas.ratio for gas in v) != 1:
            raise ValueError("The sum of the gas ratios must be 1.")

    @field_validator("thickness")
    @classmethod
    def check_thickness(cls, v):
        if v <= 0:
            raise ValueError("Gap thickness must be greater than 0.")


class GlazingSystemBSDF(BaseModel):
    name: str
    thickness: float
    layers: List[Union[Pane, ShadingBSDF]]
    gaps: List[Gap]
    visible_front_transmittance: List[List[float]]
    visible_back_transmittance: List[List[float]]
    visible_front_reflectance: List[List[float]]
    visible_back_reflectance: List[List[float]]
    solar_front_transmittance: List[List[float]]
    solar_back_transmittance: List[List[float]]
    solar_front_reflectance: List[List[float]]
    solar_back_reflectance: List[List[float]]
    solar_front_absorptance: List[List[float]]
    solar_back_absorptance: List[List[float]]

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
        with open(out.with_suffix(".json"), "w") as f:
            json.dump(self.model_dump(), f)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]):
        """Load a glazing system from a JSON file.

        Args:
            file_path: The path to the JSON file.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def get_brtdfunc(self) -> pr.Primitive:
        """Get a BRTDfunc primitive for the glazing system."""
        if not all(isinstance(layer, Pane) for layer in self.layers):
            raise ValueError("ShadingBSDF layers found.")
        if len(self.layers) > 2:
            raise ValueError("Only double pane supported.")
        if len(self.layers) == 1:
            return self.layers[0].to_brtdfunc(name=self.name)
        else:
            return get_double_pane_primitive(self.name, self.layers)


def create_wc_gaps(gaps: List[Gap]) -> List[wc.Layers.gap]:
    """Create a list of wc gaps from a list of gaps."""
    wc_gaps = []
    for gap in gaps:
        _gas = wc.create_gas(
            [[gas.ratio, getattr(wc.PredefinedGasType, gas.gas.value.upper())] for gas in gap.gases]
        )
        _gap = wc.Layers.gap(gas=_gas, thickness=gap.thickness)
        wc_gaps.append(_gap)
    return wc_gaps


def get_default_gaps(nlayers: int, thickness:float=0.0127) -> List[Gap]:
    """Get a list of default gaps."""
    gaps = []
    for _ in range(nlayers - 1):
        _gas = Gas(gas=GasType.air, ratio=1)
        gaps.append(Gap(gases=[_gas], thickness=thickness))
    return gaps


def get_solar_photopic_results(
    layers: List[wc.ProductData], gaps: List[wc.Layers.gap]
) -> Dict[str, List[List[float]]]:
    """Get the solar and photopic results.

    Args:
        layers: A list of wc.ProductData objects.
        gaps: A list of wc.Layers.gap objects.

    Returns:
        A tuple of wc.OpticalMethodResults objects.
    """
    glzsys = wc.GlazingSystem(
        solid_layers=layers,
        gap_layers=gaps,
        width_meters=1,
        height_meters=1,
        environment=wc.nfrc_shgc_environments(),
        bsdf_hemisphere=wc.BSDFHemisphere.create(wc.BSDFBasisType.FULL),
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
    layers: List[Union[Pane, ShadingBSDF]],
    gaps: Optional[List[Gap]] = None,
) -> GlazingSystemBSDF:
    """Create a glazing system from a list of layers and gaps.

    Args:
        name: The name of the glazing system.
        layers: A list of Pane or ShadingBSDF objects.
        gaps: A list of Gap objects.

    Returns:
        A GlazingSystem object.
    """
    if gaps is None:
        gaps = get_default_gaps(len(layers))

    thickness = sum(layer.thickness for layer in layers)
    thickness += sum(gap.thickness for gap in gaps)
    results = get_solar_photopic_results(
        layers=[layer.to_wincalc() for layer in layers],
        gaps=create_wc_gaps(gaps),
    )

    return GlazingSystemBSDF(
        name=name,
        thickness=thickness,
        layers=layers,
        gaps=gaps,
        **results,
    )


def create_glazing_system_from_files(
    name: str, layers: List[Union[Path, str]], gaps: Optional[List[Gap]] = None
) -> GlazingSystemBSDF:
    """Create a glazing system from a list of layers as fils and gaps.

    Args:
        name: The name of the glazing system.
        layers: A list of Path objects.
        gaps: A list of Gap objects.

    Returns:
        A GlazingSystem object.
    """
    if gaps is None:
        gaps = get_default_gaps(len(layers))
    layer_data = []
    for path in layers:
        path = Path(path)
        product_data = None
        if path.suffix == ".json":
            product_data = Pane.from_json(path)
        elif path.suffix == ".xml":
            product_data = ShadingBSDF.from_xml(path)
        if product_data is None:
            raise ValueError("Invalid layer type")
        layer_data.append(product_data)

    return create_glazing_system(name, layer_data, gaps)


def get_double_pane_primitive(name: str, panes: List[Pane]) -> pr.Primitive:
    """Generate a BRTDfunc to represent a glazing system."""
    str_arg = []
    srf_rgb = []
    for pane in panes:
        if pane.coated_side == "back":
            srf_rgb.append(pane.glass_side_rgb)
            srf_rgb.append(pane.coated_side_rgb)
        else:
            srf_rgb.append(pane.coated_side_rgb)
            srf_rgb.append(pane.glass_side_rgb)

    # reflectance depending on viewing sides
    for color in ["red", "green", "blue"]:
        str_arg.append(
            f"if(Rdot,cr(fr({getattr(srf_rgb[3], color):.3f}),ft({getattr(panes[1].transmittance_rgb, color):.3f}),fr({getattr(srf_rgb[1], color):.3f})),"
            f"cr(fr({getattr(srf_rgb[0], color):.3f}),ft({getattr(panes[0].transmittance_rgb, color):.3f}),fr({getattr(srf_rgb[2], color):.3f})))"
        )

    # transmittance
    for color in ["red", "green", "blue"]:
        str_arg.append(
            f"ft({getattr(panes[1].transmittance_rgb, color):.3f})*ft({getattr(panes[0].transmittance_rgb, color):.3f})",
        )

    str_arg.extend(["0", "0", "0", "glaze2.cal"])
    real_arg = [.0] * 9
    return pr.Primitive("void", "BRTDfunc", name, str_arg, real_arg)


def laminate(base: Pane, lam: Pane, side: wc.CoatedSide) -> Pane:
    """Laminate a glazing layer."""
    if lam.substrate is None:
        raise ValueError("Laminate must have substrate data")
    # deconstruct laminate
    optical_data = wc.ProductDataOpticalNBand()
    layer = wc.ProductDataOpticalAndThermal()
    ...
