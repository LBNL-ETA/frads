from pathlib import Path
from typing import Tuple

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
    # default_air_gap = pwc.Gap(AIR, 0.0127)
    default_air_gap = (AIR, 1), 0.0127

    def __init__(self):
        self._name = ""
        self._gaps = []
        self.layers = []
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
    def gaps(self, value: Tuple[Tuple[pwc.PredefinedGasType, float], float]):
        """Set the gaps."""
        self._gaps = value
        self.updated = True

    def add_glazing_layer(self, inp):
        """Add a glazing layer."""
        if isinstance(inp, (str, Path)):
            _path = Path(inp)
            if not _path.exists():
                raise FileNotFoundError(inp)
            if _path.suffix == "json":
                data = pwc.parse_json_file(str(_path))
            else:
                data = pwc.parse_optics_file(str(_path))
        else:
            data = pwc.parse_json(inp)
        self.layers.append(data)
        if len(self.layers) > 1:
            self._gaps.append(self.default_air_gap)

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
        if len(self.layers) > 1:
            self._gaps.append(self.default_air_gap)

    def build(self):
        """Build the glazing system."""
        if (len(self.layers) - 1) != len(self.gaps):
            raise ValueError("Number of gaps must be one less than number of layers.")
        self.glzsys = pwc.GlazingSystem(
            optical_standard=pwc.load_standard(
                str(Path(__file__).parent / "data" / "optical_standards" / "W5_NFRC_2003.std")
            ),
            solid_layers=self.layers,
            gap_layers=[create_gap(*g[:-1], thickness=g[1]) for g in self.gaps],
            width_meters=1,
            height_meters=1,
            environment=pwc.nfrc_shgc_environments(),
            bsdf_hemisphere=pwc.BSDFHemisphere.create(pwc.BSDFBasisType.FULL),
        )

    def compute_solar_photopic_results(self, force=False):
        """Compute the solar photopic results."""
        # compute = False
        if None not in (self.solar_results , self.photopic_results):
            if self.layers != self.glzsys.solid_layers or self.gaps != self.glzsys.gap_layers:
                self.updated = True
        self.updated = True if force else self.updated
        if self.updated:
            self.build()
            self.solar_results = self.glzsys.optical_method_results("SOLAR")
            self.photopic_results = self.glzsys.optical_method_results("PHOTOPIC")
