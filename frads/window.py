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
        self.layers = []
        self.gaps = []
        self._name = ""
        self.solar_results = None
        self.photopic_results = None

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
            self.gaps.append(self.default_air_gap)

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
            self.gaps.append(self.default_air_gap)


    def compute_solar_photopic_results(self):
        """Compute the solar photopic results."""
        if (len(self.layers) - 1) != len(self.gaps):
            raise ValueError("Number of gaps must be one less than number of layers.")
        gs = pwc.GlazingSystem(
            optical_standard=pwc.load_standard(
                str(Path(__file__).parent / "data" / "optical_standards" / "W5_NFRC_2003.std")
            ),
            solid_layers=self.layers,
            gap_layers=[create_gap(g[0], thickness=g[1]) for g in self.gaps],
            width_meters=1,
            height_meters=1,
            environment=pwc.nfrc_shgc_environments(),
            bsdf_hemisphere=pwc.BSDFHemisphere.create(pwc.BSDFBasisType.FULL),
        )
        self.solar_results = gs.optical_method_results("SOLAR")
        self.photopic_results = gs.optical_method_results("PHOTOPIC")


def add_cfs_to_epjs(glazing_system, epjs) -> None:
    """
    Add CFS to an EnergyPlus JSON file.

    Args:
        solar_results: Solar results from pywincalc.
        photopic_results: Photopic results from pywincalc.
        glazing_system: Glazing system from pywincalc.
        epjs: EnergyPlus JSON file.
    Returns:
        None
    """
    name = glazing_system.name
    if glazing_system.solar_results is not None and glazing_system.photopic_results is not None:
        solar_results = glazing_system.solar_results
        photopic_results = glazing_system.photopic_results
    else:
        raise ValueError("Solar and photopic results must be computed first.")
    

    # Initialize Contruction:ComplexFenestrationState dictionary with system and outer layer names

    construction_complex_fenestration_state = {}

    construction_complex_fenestration_state[name] = {
        "basis_matrix_name": f"{name}_Basis",
        "basis_symmetry_type": "None",
        "basis_type": "LBNLWINDOW",
        "solar_optical_complex_back_reflectance_matrix_name": f"{name}_RbSol",
        "solar_optical_complex_front_transmittance_matrix_name": f"{name}_TfSol",
        "visible_optical_complex_back_transmittance_matrix_name": f"{name}_Tbvis",
        "visible_optical_complex_front_transmittance_matrix_name": f"{name}_Tfvis",
        "window_thermal_model": "ThermParam_1",
        "outside_layer_directional_back_absoptance_matrix_name": f"{name}_layer_1_bAbs",
        "outside_layer_directional_front_absoptance_matrix_name": f"{name}_layer_1_fAbs",
        "outside_layer_name": glazing_system.layers[0].product_name,
    }

    # Initialize Matrix:TwoDimension dictionary with system and outer layer matrices
    matrix_two_dimension = {
        construction_complex_fenestration_state[name]["basis_matrix_name"]: {
            "number_of_columns": 2,
            "number_of_rows": 9,
            "values": [
                {"value": 0.0},
                {"value": 1.0},
                {"value": 10.0},
                {"value": 8.0},
                {"value": 20.0},
                {"value": 16.0},
                {"value": 30.0},
                {"value": 20.0},
                {"value": 40.0},
                {"value": 24.0},
                {"value": 50.0},
                {"value": 24.0},
                {"value": 60.0},
                {"value": 24.0},
                {"value": 70.0},
                {"value": 16.0},
                {"value": 82.5},
                {"value": 12.0},
            ],
        },
        construction_complex_fenestration_state[name]["solar_optical_complex_back_reflectance_matrix_name"]: {
            "number_of_columns": 145,
            "number_of_rows": 145,
            "values": [
                {"value": val}
                for row in solar_results.system_results.back.reflectance.matrix for val in row
            ],
        },
        construction_complex_fenestration_state[name]["solar_optical_complex_front_transmittance_matrix_name" ]: {
            "number_of_columns": 145,
            "number_of_rows": 145,
            "values": [
                {"value": val}
                for row in solar_results.system_results.front.transmittance.matrix for val in row
            ],
        },
        construction_complex_fenestration_state[name]["visible_optical_complex_back_transmittance_matrix_name"]: {
            "number_of_columns": 145,
            "number_of_rows": 145,
            "values": [
                {"value": val}
                for row in photopic_results.system_results.back.transmittance.matrix
                for val in row
            ],
        },
        construction_complex_fenestration_state[name][
            "visible_optical_complex_front_transmittance_matrix_name"
        ]: {
            "number_of_columns": 145,
            "number_of_rows": 145,
            "values": [
                {"value": val}
                for row in photopic_results.system_results.front.transmittance.matrix
                for val in row
            ],
        },
        construction_complex_fenestration_state[name][
            "outside_layer_directional_back_absoptance_matrix_name"
        ]: {
            "number_of_columns": 145,
            "number_of_rows": 1,
            "values": [
                {"value": val}
                for val in solar_results.layer_results[0].back.absorptance.angular_total
            ],
        },
        construction_complex_fenestration_state[name][
            "outside_layer_directional_front_absoptance_matrix_name"
        ]: {
            "number_of_columns": 145,
            "number_of_rows": 1,
            "values": [
                {"value": val}
                for val in solar_results.layer_results[
                    0
                ].front.absorptance.angular_total
            ],
        },
    }

    # Define layer absorptance names and matrices for the rest of the layers.
    for i in range(len(glazing_system.layers) - 1):
        _layer_name = f"{name}_layer_{i+2}"
        construction_complex_fenestration_state[name][
            f"layer_{i+2}_directional_back_absoptance_matrix_name"
        ] = f"{_layer_name}_bAbs"
        matrix_two_dimension[f"{_layer_name}_bAbs"] = {
            "number_of_columns": 145,
            "number_of_rows": 1,
            "values": [
                {"value": val}
                for val in solar_results.layer_results[
                    i + 1
                ].back.absorptance.angular_total
            ],
        }
        construction_complex_fenestration_state[name][
            f"layer_{i+2}_directional_front_absoptance_matrix_name"
        ] = f"{_layer_name}_fAbs"
        matrix_two_dimension[f"{_layer_name}_fAbs"] = {
            "number_of_columns": 145,
            "number_of_rows": 1,
            "values": [
                {"value": val}
                for val in solar_results.layer_results[
                    i + 1
                ].front.absorptance.angular_total
            ],
        }
        construction_complex_fenestration_state[name][
            f"layer_{i+2}_name"
        ] = glazing_system.layers[i + 1].product_name

    # Define gap and gas layer
    standard_atmosphere_pressure = 101325.0
    window_material_gap = {}
    window_material_gas = {}
    for i, gap in enumerate(glazing_system.gaps, 1):
        _gap_name = f"{name}_gap_{i}"
        construction_complex_fenestration_state[name][
            f"gap_{i}_name"
        ] = f"{_gap_name}_layer"
        _gas_name = f"gas_{i}"
        window_material_gap[f"{_gap_name}_layer"] = {
            "gas_or_gas_mixture_": _gas_name,
            "pressure": standard_atmosphere_pressure,
            "thickness": gap[1],
        }
        window_material_gas[_gas_name] = {
            "gas_type": gap[0][0].name.capitalize(),
            "thickness": gap[1],
        }

    # Define glazing and shading layer
    window_material_glazing = {}
    window_material_complex_shade = {}

    for layer in glazing_system.layers:
        # glazing
        if layer.product_type == "glazing":
            window_material_glazing[layer.product_name] = {
                "back_side_infrared_hemispherical_emissivity": layer.emissivity_back,
                "conductivity": layer.conductivity,
                "front_side_infrared_hemispherical_emissivity": layer.emissivity_front,
                "infrared_transmittance_at_normal_incidence": layer.ir_transmittance,
                "optical_data_type": "BSDF",
                "poisson_s_ratio": 0.22,
                "thickness": layer.thickness,
                "window_glass_spectral_data_set_name": "",
                # "young_s_modulus": layer.youngs_modulus,
            }
        # Assuming complex shade if not glazing
        else:
            window_material_complex_shade[layer.product_name] = {
                "back_emissivity": layer.emissivity_back,
                "top_opening_multiplier": 0,
                "bottom_opening_multiplier": 0,
                "left_side_opening_multiplier": 0,
                "right_side_opening_multiplier": 0,
                "front_opening_multiplier": 0.05,
                "conductivity": layer.conductivity,
                "front_emissivity": layer.emissivity_front,
                "ir_transmittance": layer.ir_transmittance,
                "layer_type": "BSDF",
                "thickness": layer.thickness,
            }

    window_thermal_model_params = {
        "ThermParam_1": {
            "deflection_model": "NoDeflection",
            "sdscalar": 1.0,
            "standard": "ISO15099",
            "thermal_model": "ISO15099",
        }
    }

    mappings = {
        "Construction:ComplexFenestrationState": construction_complex_fenestration_state,
        "Matrix:TwoDimension": matrix_two_dimension,
        "WindowMaterial:Gas": window_material_gas,
        "WindowMaterial:Gap": window_material_gap,
        "WindowMaterial:Glazing": window_material_glazing,
        "WindowMaterial:ComplexShade": window_material_complex_shade,
        "WindowThermalModel:Params": window_thermal_model_params,
    }

    for key, val in mappings.items():
        if key in epjs:
            epjs[key] = {**epjs[key], **val}
        else:
            epjs[key] = val

    # Set the all fenestration surface constructions to complex fenestration state
    # pick the first cfs
        cfs = list(epjs['Construction:ComplexFenestrationState'].keys())[0]
        for window_name in epjs["FenestrationSurface:Detailed"]:
            epjs["FenestrationSurface:Detailed"][window_name][
                "construction_name"
            ] = cfs


def add_lighting_epjs(epjs):
    """Add lighting objects to the epjs dictionary."""

    # Initialize lighting schedule type limit dictionary
    schedule_type_limit = {} 
    schedule_type_limit["on_off"] = {
        "lower_limit_value" : 0,
        "upper_limit_value" : 1,
        "numeric_type" : "Discrete",
        "unit_type" : "Availability"
    }

    # Initialize lighting schedule dictionary
    lighting_schedule = {}
    lighting_schedule["constant_off"] = {
        "schedule_type_limits_name" : "on_off",
        "hourly_value" : 0
    }

    # Initialize lights dictionary with a constant-off schedule for each zone

    lights = {}
    for zone in epjs["Zone"]:
        _name = f"Light_{zone}"
        lights[_name] = {
            "design_level_calculation_method": "LightingLevel",
            "fraction_radiant": 0,
            "fraction_replaceable": 1,
            "fraction_visible": 1,
            "lighting_level": 0,
            "return_air_fraction": 0,
            "schedule_name": "constant_off",
            "zone_or_zonelist_or_space_or_spacelist_name": zone,
        }

    # Add lighting output to the epjs dictionary

    i = 1
    for output in epjs["Output:Variable"].values():
        i += 1
        if output["variable_name"] == "Lights Electricity Rate":
            break
    else:
        epjs["Output:Variable"][f"Output:Variable {i}"] = {
            "key_value": "*",
            "reporting_frequency": "Timestep",
            "variable_name": "Lights Electricity Rate",
        }

    mappings = {
        "ScheduleTypeLimits" : schedule_type_limit,
        "Schedule:Constant" : lighting_schedule,
        "Lights" : lights
    }

    for key, val in mappings.items():
        if key in epjs:
            epjs[key] = {**epjs[key], **val}
        else:
            epjs[key] = val
