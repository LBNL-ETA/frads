import tempfile
from pathlib import Path
from typing import List, Tuple

import pyradiance as pr
import pywincalc as pwc

AIR = pwc.PredefinedGasType.AIR
KRYPTON = pwc.PredefinedGasType.KRYPTON
XENON = pwc.PredefinedGasType.XENON
ARGON = pwc.PredefinedGasType.ARGON


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
