import tempfile
from pathlib import Path
from typing import List, NamedTuple, Tuple

import pyradiance as pr
import pywincalc as pwc

AIR = pwc.PredefinedGasType.AIR
KRYPTON = pwc.PredefinedGasType.KRYPTON
XENON = pwc.PredefinedGasType.XENON
ARGON = pwc.PredefinedGasType.ARGON


class PaneRGB(NamedTuple):
    """Pane color data object.

    Attributes:
        measured_data: measured data as a PaneProperty object.
        coated_rgb: Coated side RGB.
        glass_rgb: Non-coated side RGB.
        trans_rgb: Transmittance RGB.
    """

    measured_data: pwc.ProductData
    coated_rgb: Tuple[float, float, float]
    glass_rgb: Tuple[float, float, float]
    trans_rgb: Tuple[float, float, float]


def create_gap(*gases_ratios: Tuple[pwc.PredefinedGasType, float], thickness):
    """Create a gap with the gas and thickness."""
    if len(gases_ratios) > 1:
        if sum([ratio for _, ratio in gases_ratios]) != 1:
            raise ValueError("The sum of the gas ratios must be 1.")
        components = [
            pwc.PredefinedGasMixtureComponent(gas, ratio) for gas, ratio in gases_ratios
        ]
        return pwc.Gap(components, thickness)
    return pwc.Gap(gases_ratios[0][0], thickness)


# class Layer:
#     def __init__(self, inp):
#         self.inp = inp
#         if isinstance(inp, (str, Path)):
#             self.inp = Path(inp)
#             if not self.inp.exists():
#                 raise FileNotFoundError(inp)
#         self.data = None
#         self.thickness = 0
#
#
# class Glazing(Layer):
#     def __init__(self, inp, name=None):
#         super().__init__(inp)
#         if isinstance(self.inp, Path):
#             product_name = self.inp.stem
#             if self.inp.suffix == ".json":
#                 self.data = pwc.parse_json_file(str(self.inp))
#             else:
#                 self.data = pwc.parse_optics_file(str(self.inp))
#         else:
#             self.data = pwc.parse_json(self.inp)
#             product_name = self.inp["name"] or self.inp["product_name"]
#         self.data.product_name = (
#             self.data.product_name or product_name or name or str(inp)[:6]
#         )
#         self.thickness = self.data.thickness
#
#
# class Shading(Layer):
#     def __init__(self, inp, name=None):
#         super().__init__(inp)
#         if isinstance(self.inp, Path):
#             self.data = pwc.parse_bsdf_xml_file(str(self.inp))
#         else:
#             self.data = pwc.parse_bsdf_xml_string(self.inp)
#         self.thickness = self.data.thickness
#         self.data.product_name = self.data.product_name or name or str(inp)[:6]
#
#
# class AppliedFilm(Glazing):
#     def __init__(self, inp, name=None):
#         super().__init__(inp, name=name)


# class Gap(Layer):
#     def __init__(self, *gases_ratios, thickness):
#         if len(gases_ratios) > 1:
#             if sum([ratio for _, ratio in gases_ratios]) != 1:
#                 raise ValueError("The sum of the gas ratios must be 1.")
#             components = [
#                 pwc.PredefinedGasMixtureComponent(gas, ratio)
#                 for gas, ratio in gases_ratios
#             ]
#             self.data = pwc.Gap(components, thickness)
#         self.data = pwc.Gap(gases_ratios[0][0], thickness)


class GlazingSystem:
    default_air_gap = (AIR, 1), 0.0127

    def __init__(self):
        self._name = ""
        self._gaps = []
        self.layers = []
        self._thickness = 0
        self.glzsys = None
        self.photopic_results = None
        self.solar_results = None
        self.updated = True

    @classmethod
    def from_gls(cls, gls_path):
        """Create a GlazingSystem from a glazing system file."""
        # unzip the gls file
        pass

    @property
    def name(self):
        """Return the name of the glazing system."""
        if self._name:
            return self._name
        return "_".join([l.product_name for l in self.layers])

    @name.setter
    def name(self, value):
        """Set the name of the glazing system."""
        self._name = value

    @property
    def gaps(self):
        """Return the gaps."""
        return self._gaps

    @gaps.setter
    def gaps(self, value: List[Tuple[Tuple[pwc.PredefinedGasType, float], float]]):
        """Set the gaps."""
        self._gaps = value
        self._thickness -= len(value) * self.default_air_gap[-1]
        self._thickness += sum([g[-1] for g in value])
        self.updated = True

    def add_glazing_layer(self, inp):
        """Add a glazing layer."""
        if isinstance(inp, (str, Path)):
            _path = Path(inp)
            product_name = _path.stem
            if not _path.exists():
                raise FileNotFoundError(inp)
            if _path.suffix == ".json":
                data = pwc.parse_json_file(str(_path))
            else:
                data = pwc.parse_optics_file(str(_path))
        else:
            data = pwc.parse_json(inp)
            product_name = inp["name"] or inp["product_name"]
        data.product_name = data.product_name or product_name or str(inp)[:6]
        self.layers.append(data)

        self._thickness += data.thickness / 1000.0 or 0  # mm to m
        if len(self.layers) > 1:
            self._gaps.append(self.default_air_gap)
            self._thickness += self.default_air_gap[-1]
        self.updated = True

    def add_shading_layer(self, inp):
        """Add a shading layer."""
        if isinstance(inp, (str, Path)):
            _path = Path(inp)
            if not _path.exists():
                raise FileNotFoundError(inp)
            data = pwc.parse_bsdf_xml_file(str(_path))
        else:
            data = pwc.parse_bsdf_xml_string(inp)
        self.layers.append(data)
        self._thickness += data.thickness / 1e3 or 0
        if len(self.layers) > 1:
            self._gaps.append(self.default_air_gap)
            self._thickness += self.default_air_gap[-1]
        self.updated = True

    # def add_film_layer(self, inp, glazing, inside=False):
    #     """Add a film layer."""
    #     film = AppliedFilm(inp)

    #     if isinstance(inp, (str, Path)):
    #         _path = Path(inp)
    #         if not _path.exists():
    #             raise FileNotFoundError(inp)
    #         data = pwc.parse_optics_file(str(_path))
    #     else:
    #         data = pwc.parse_json(inp)
    #     if inside:
    #         self.layers.append(data)
    #     else:
    #         self.layers.insert(0, data)
    #     self._thickness += data.thickness / 1e3 or 0
    #     self.updated = True

    def build(self):
        """Build the glazing system."""
        if (len(self.layers) - 1) != len(self.gaps):
            raise ValueError("Number of gaps must be one less than number of layers.")

        self.glzsys = pwc.GlazingSystem(
            solid_layers=self.layers,
            gap_layers=[create_gap(*g[:-1], thickness=g[-1]) for g in self._gaps],
            width_meters=1,
            height_meters=1,
            environment=pwc.nfrc_shgc_environments(),
            bsdf_hemisphere=pwc.BSDFHemisphere.create(pwc.BSDFBasisType.FULL),
        )

    def compute_solar_photopic_results(self, force=False):
        """Compute the solar photopic results."""
        self.updated = True if force else self.updated
        if self.updated:
            self.build()
            self.solar_results = self.glzsys.optical_method_results("SOLAR")
            self.photopic_results = self.glzsys.optical_method_results("PHOTOPIC")
            self.updated = False

    def to_xml(self, out):
        """Save the glazing system to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _tbv = Path(tmpdir) / "tbv"
            _tfv = Path(tmpdir) / "tfv"
            _rfv = Path(tmpdir) / "rfv"
            _rbv = Path(tmpdir) / "rbv"
            _tbs = Path(tmpdir) / "tbs"
            _tfs = Path(tmpdir) / "tfs"
            _rfs = Path(tmpdir) / "rfs"
            _rbs = Path(tmpdir) / "rbs"
            with open(_tbv, "w") as f:
                f.write(
                    "\n".join(
                        " ".join(str(n) for n in row)
                        for row in self.photopic_results.system_results.front.transmittance.matrix
                    )
                )
            with open(_tfv, "w") as f:
                f.write(
                    "\n".join(
                        " ".join(str(n) for n in row)
                        for row in self.photopic_results.system_results.back.transmittance.matrix
                    )
                )
            with open(_rfv, "w") as f:
                f.write(
                    "\n".join(
                        " ".join(str(n) for n in row)
                        for row in self.photopic_results.system_results.front.reflectance.matrix
                    )
                )
            with open(_rbv, "w") as f:
                f.write(
                    "\n".join(
                        " ".join(str(n) for n in row)
                        for row in self.photopic_results.system_results.back.reflectance.matrix
                    )
                )
            with open(_tbs, "w") as f:
                f.write(
                    "\n".join(
                        " ".join(str(n) for n in row)
                        for row in self.solar_results.system_results.front.transmittance.matrix
                    )
                )
            with open(_tfs, "w") as f:
                f.write(
                    "\n".join(
                        " ".join(str(n) for n in row)
                        for row in self.solar_results.system_results.back.transmittance.matrix
                    )
                )
            with open(_rfs, "w") as f:
                f.write(
                    "\n".join(
                        " ".join(str(n) for n in row)
                        for row in self.solar_results.system_results.front.reflectance.matrix
                    )
                )
            with open(_rbs, "w") as f:
                f.write(
                    "\n".join(
                        " ".join(str(n) for n in row)
                        for row in self.solar_results.system_results.back.reflectance.matrix
                    )
                )
            _vi = pr.WrapBSDFInput("Visible", _tbv, _tfv, _rfv, _rbv)
            _si = pr.WrapBSDFInput("Solar", _tbs, _tfs, _rfs, _rbs)
            with open(out, "wb") as f:
                f.write(
                    pr.wrapbsdf(
                        enforce_window=True,
                        basis="kf",
                        inp=[_vi, _si],
                        n=self.name,
                        m="",
                        t=str(self._thickness),
                    )
                )

    def save(self):
        """
        Compress the glazing system into a .gls file.
        A .gls file contain individual layer data and gap data.
        System matrix results are also included.
        """
        pass

    def gen_glazing(self):
        """
        Generate a brtdfunc for a single or double pane glazing system.
        """
        # Check if is more than two layers
        if len(self.layers) > 2:
            raise ValueError("Only single and double pane supported")
        # Check if all layers are glazing
        for layer in self.layers:
            if not layer.type == "glazing":
                raise ValueError("Only glazing layers supported")
        # Call gen_glaze to generate brtdfunc
        return


def get_glazing_primitive(panes: List[PaneRGB]) -> pr.Primitive:
    """Generate a BRTDfunc to represent a glazing system."""
    if len(panes) > 2:
        raise ValueError("Only double pane supported")
    names = []
    for pane in panes:
        names.append(pane.measured_data.product_name or "Unnamed")
    name = "+".join(names)
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
        coated_real = 1 if panes[0].measured_data.coated_side == "front" else -1
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
        if panes[0].measured_data.coated_side == "back":
            s2r_r, s2r_g, s2r_b = panes[0].coated_rgb
            s1r_r, s1r_g, s1r_b = panes[0].glass_rgb
        else:  # front or neither side coated
            s2r_r, s2r_g, s2r_b = panes[0].glass_rgb
            s1r_r, s1r_g, s1r_b = panes[0].coated_rgb
        if panes[1].measured_data.coated_side == "back":
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
