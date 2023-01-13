import datetime
import json
from pathlib import Path
from typing import Optional


def ep_datetime_parser(inp):
    date, time = inp.strip().split()
    month, day = [int(i) for i in date.split("/")]
    hr, mi, sc = [int(i) for i in time.split(":")]
    if hr == 24 and mi == 0 and sc == 0:
        return datetime.datetime(1900, month, day, 0, mi, sc) + datetime.timedelta(
            days=1
        )
    else:
        return datetime.datetime(1900, month, day, hr, mi, sc)


def load_epmodel(fpath: Path, api) -> dict:
    """Load and parse input file into a JSON object.
    If the input file is in .idf fomart, use command-line
    energyplus program to convert it to epJSON format
    Args:
        fpath: input file path
    Returns:
        epjs: JSON object as a Python dictionary
    """
    epjson_path: Path
    if fpath.suffix == ".idf":
        state = api.state_manager.new_state()
        api.runtime.set_console_output_status(state, False)
        api.runtime.run_energyplus(state, ["--convert-only", str(fpath)])
        api.state_manager.delete_state(state)
        epjson_path = Path(fpath.with_suffix(".epJSON").name)
        if not epjson_path.is_file():
            raise FileNotFoundError(f"Converted {str(epjson_path)} not found.")
    elif fpath.suffix == ".epJSON":
        epjson_path = fpath
    else:
        raise Exception(f"Unknown file type {fpath}")
    with open(epjson_path) as rdr:
        epjs = json.load(rdr)

    return EPModel(epjs)


class Handles:
    def __init__(self):
        self.outdoor_drybulb_temperature = None
        self.direct_normal_irradiance = None
        self.diffuse_horizontal_irradiance = None
        self.complex_fenestration_state = {}
        self.window_actuators = {}
        self.lights_electricity_energy = {}
        self.light_actuators = {}


class EnergyPlusSetup:
    def __init__(self, api, epjs):
        self.api = api
        self.epjs = epjs
        self.state = self.api.state_manager.new_state()
        self.handles = Handles()
        self.window_surfaces = self.epjs["FenestrationSurface:Detailed"]
        self.lights = self.epjs["Lights"]
        self.api.exchange.request_variable(
            self.state,
            "Site Outdoor Air Drybulb Temperature".encode(),
            "Environment".encode(),
        )
        self.api.exchange.request_variable(
            self.state,
            "Site Direct Solar Radiation Rate per Area".encode(),
            "Environment".encode(),
        )
        self.api.exchange.request_variable(
            self.state,
            "Site Diffuse Solar Radiation Rate per Area".encode(),
            "Environment".encode(),
        )
        for light in self.lights:
            self.api.exchange.request_variable(
                self.state, "Lights Total Heating Rate".encode(), light.encode()
            )

        self.api.runtime.callback_begin_new_environment(self.state, self.get_handles())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.api.state_manager.delete_state(self.state)

    def actuate(self, obj, value):
        self.api.exchange.set_actuator_value(self.state, obj, value)

    def get_variable_value(self, handle):
        return self.api.exchange.get_variable_value(self.state, handle)

    def get_variable_handle(self, variable_name, variable_key):
        return self.api.exchange.get_variable_handle(
            self.state, variable_name, variable_key
        )

    def get_handles(self):
        def callback_function(state):
            self.handles.outdoor_drybulb_temperature = (
                self.api.exchange.get_variable_handle(
                    state, "Site Outdoor Air Drybulb Temperature", "Environment"
                )
            )
            self.handles.direct_normal_irradiance = (
                self.api.exchange.get_variable_handle(
                    state, "Site Direct Solar Radiation Rate per Area", "Environment"
                )
            )
            self.handles.diffuse_horizontal_irradiance = (
                self.api.exchange.get_variable_handle(
                    state, "Site Diffuse Solar Radiation Rate per Area", "Environment"
                )
            )
            construction_complex_fenestration_state = self.epjs[
                "Construction:ComplexFenestrationState"
            ]

            for cfs in construction_complex_fenestration_state:
                self.handles.complex_fenestration_state[
                    cfs
                ] = self.api.api.getConstructionHandle(state, cfs.encode())
            for window in self.window_surfaces:
                self.handles.window_actuators[
                    window
                ] = self.api.exchange.get_actuator_handle(
                    state, "Surface", "Construction State", window.encode()
                )

            for light in self.lights:
                self.handles.lights_electricity_energy[
                    light
                ] = self.api.exchange.get_variable_handle(
                    state, "Lights Electricity Energy", light.encode()
                )
                self.handles.light_actuators[
                    light
                ] = self.api.exchange.get_actuator_handle(
                    state, "Lights", "Electricity Rate", light.encode()
                )

        return callback_function

    def get_datetime(self):
        year = self.api.exchange.year(self.state)
        month = self.api.exchange.month(self.state)
        day = self.api.exchange.day_of_month(self.state)
        hour = self.api.exchange.hour(self.state)
        minute = self.api.exchange.minutes(self.state)

        if minute == 60:
            minute = 0
            hour += 1
        if hour == 24:
            hour = 0
            day += 1
        if day == 31 and month in [4, 6, 9, 11]:
            day = 1
            month += 1
        if day == 32 and month in [1, 3, 5, 7, 8, 10, 12]:
            day = 1
            month += 1
        if year % 4 == 0 and day == 30 and month == 2:
            day = 1
            month += 1
        else:
            if day == 29 and month == 2:
                day = 1
                month += 1

        dt = datetime.datetime(year, month, day, hour, minute)

        return dt

    def run(
        self,
        weather_file: Optional[str] = None,
        output_directory: Optional[str] = None,
        output_prefix: Optional[str] = "eplus",
    ):

        options = {"-w": weather_file, "-d": output_directory, "-p": output_prefix}
        # check if any of options are None, if so, dont pass them to run_energyplus
        options = {k: v for k, v in options.items() if v is not None}
        opt = [item for sublist in options.items() for item in sublist]

        with open(f"{output_prefix}.json", "w") as wtr:
            json.dump(self.epjs, wtr)

        self.api.runtime.run_energyplus(
            self.state, [*opt, "-r", f"{output_prefix}.json"]
        )

    def set_callback(self, method_name: str, func):
        try:
            method = getattr(self.api.runtime, method_name)
        except AttributeError:
            raise AttributeError(
                f"Method {method_name} not found in EnergyPlus runtime API"
            )
        # method(self.state, partial(func, self))
        method(self.state, func)


class EPModel:
    def __init__(self, epjs):
        self.epjs = epjs

    @property
    def cfs(self):
        """
        Example:
            >>> model.cfs
        """
        return list(self.epjs["Construction:ComplexFenestrationState"].keys())

    @property
    def windows(self):
        """
        Example:
            >>> model.windows
        """
        return list(self.epjs["FenestrationSurface:Detailed"].keys())

    @property
    def lighting_zone(self):
        return list(self.epjs["Lights"].keys())

    @property
    def zones(self):
        return list(self.epjs["Zone"].keys())

    def _add(self, key, obj):
        if key in self.epjs:
            # merge
            self.epjs[key] = {**self.epjs[key], **obj}
        else:
            # add
            self.epjs[key] = obj

    def add_cfs(self, glazing_system) -> None:
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
        if (
            glazing_system.solar_results is not None
            and glazing_system.photopic_results is not None
        ):
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
            construction_complex_fenestration_state[name][
                "solar_optical_complex_back_reflectance_matrix_name"
            ]: {
                "number_of_columns": 145,
                "number_of_rows": 145,
                "values": [
                    {"value": val}
                    for row in solar_results.system_results.back.reflectance.matrix
                    for val in row
                ],
            },
            construction_complex_fenestration_state[name][
                "solar_optical_complex_front_transmittance_matrix_name"
            ]: {
                "number_of_columns": 145,
                "number_of_rows": 145,
                "values": [
                    {"value": val}
                    for row in solar_results.system_results.front.transmittance.matrix
                    for val in row
                ],
            },
            construction_complex_fenestration_state[name][
                "visible_optical_complex_back_transmittance_matrix_name"
            ]: {
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
                    for val in solar_results.layer_results[
                        0
                    ].back.absorptance.angular_total
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

        for key, obj in mappings.items():
            self._add(key, obj)

            # Set the all fenestration surface constructions to complex fenestration state
            # pick the first cfs
            cfs = list(self.epjs["Construction:ComplexFenestrationState"].keys())[0]
            for window_name in self.epjs["FenestrationSurface:Detailed"]:
                self.epjs["FenestrationSurface:Detailed"][window_name][
                    "construction_name"
                ] = cfs

    def add_lighting(self):
        """Add lighting objects to the epjs dictionary."""

        # Initialize lighting schedule type limit dictionary
        schedule_type_limit = {}
        schedule_type_limit["on_off"] = {
            "lower_limit_value": 0,
            "upper_limit_value": 1,
            "numeric_type": "Discrete",
            "unit_type": "Availability",
        }

        # Initialize lighting schedule dictionary
        lighting_schedule = {}
        lighting_schedule["constant_off"] = {
            "schedule_type_limits_name": "on_off",
            "hourly_value": 0,
        }

        # Initialize lights dictionary with a constant-off schedule for each zone

        lights = {}
        for zone in self.epjs["Zone"]:
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
        for output in self.epjs["Output:Variable"].values():
            i += 1
            if output["variable_name"] == "Lights Electricity Rate":
                break
        else:
            self.epjs["Output:Variable"][f"Output:Variable {i}"] = {
                "key_value": "*",
                "reporting_frequency": "Timestep",
                "variable_name": "Lights Electricity Rate",
            }

        mappings = {
            "ScheduleTypeLimits": schedule_type_limit,
            "Schedule:Constant": lighting_schedule,
            "Lights": lights,
        }

        for key, obj in mappings.items():
            self._add(key, obj)


# def shade_controller(ep: EnergyPlusSetup, state) -> None:
#     """
#     This is a user implemented controller, which gets called at EnergyPlus runtime.
#     Args:
#         ep: EnergyPlusSetup object
#         state: EnergyPlus state
#     Returns:
#         None
#     """
#     drybulb = ep.get_variable_value(ep.handles.outdoor_drybulb_temperature)
#     dni = ep.get_variable_value(ep.handles.direct_normal_irradiance)
#     dhi = ep.get_variable_value(ep.handles.diffuse_horizontal_irradiance)
#     if dni > 800:
#         ep.actuate(ep.handles.window_actuators['Window1'], ep.handles.complex_fenestration_state['Window1'])
#
#
# with EnergyPlusSetup(api, epjs) as ep:
#     ep.set_callback("callback_begin_new_environment", shade_controller)
#     ep.run()
