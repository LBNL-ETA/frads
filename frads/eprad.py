"""
Class and functions for accessing EnergyPlus Python API
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Optional, Callable, Union

from frads import sky
from frads.window import GlazingSystem
import copy
from pyenergyplus.api import EnergyPlusAPI


class EnergyPlusModel:
    """EnergyPlus Model object

    Attributes:
        api: EnergyPlus runtime API object
        epjs: EnergyPlus JSON object
        actuators_list: list of actuators available for this model
        cfs: list of complex fenestration states
        windows: list of windows
        walls_window: list of walls with windows
        floors: list of floors
        lighting_zone: list of lighting zones
        zones: list of zones

    If the input file is in .idf format, use command-line EnergyPlus program \
        to convert it to epJSON format

    Example:
        >>> model = EnergyPlusModel(Path("model.idf"))
    """

    def __init__(self, fpath: Union[str, Path]):
        """Load and parse input file into a JSON object.
        If the input file is in .idf format, use command-line
        energyplus program to convert it to epJSON format

        Args:
            fpath: input file path
        """
        self.api = EnergyPlusAPI()
        fpath = Path(fpath)
        epjson_path: Path
        if fpath.suffix == ".idf":
            state = self.api.state_manager.new_state()
            self.api.runtime.set_console_output_status(state, False)
            self.api.runtime.run_energyplus(state, ["--convert-only", str(fpath)])
            self.api.state_manager.delete_state(state)
            epjson_path = Path(fpath.with_suffix(".epJSON").name)
            if not epjson_path.is_file():
                raise FileNotFoundError(f"Converted {str(epjson_path)} not found.")
        elif fpath.suffix == ".epJSON":
            epjson_path = fpath
        else:
            raise Exception(f"Unknown file type {fpath}")
        with open(epjson_path) as rdr:
            self.epjs = json.load(rdr)

    @property
    def complex_fenestration_states(self):
        """
        Example:
            >>> model.complex_fenestration_states
        """
        if "Construction:ComplexFenestrationState" in self.epjs:
            return list(self.epjs["Construction:ComplexFenestrationState"].keys())
        return []

    @property
    def windows(self):
        """
        Example:
            >>> model.windows
        """
        if "FenestrationSurface:Detailed" in self.epjs:
            return list(self.epjs["FenestrationSurface:Detailed"].keys())
        return []

    @property
    def window_walls(self):
        """
        Example:
            >>> model.window_walls
        """
        wndo_walls = []
        if "FenestrationSurface:Detailed" in self.epjs:
            for k, v in self.epjs["FenestrationSurface:Detailed"].items():
                wndo_walls.append(v["building_surface_name"])
        return wndo_walls

    @property
    def floors(self):
        """
        Example:
            >>> model.floors
        """
        floors = []
        if "BuildingSurface:Detailed" in self.epjs:
            for k, v in self.epjs["BuildingSurface:Detailed"].items():
                if v["surface_type"] == "Floor":
                    floors.append(k)
        return floors

    @property
    def lights(self):
        """
        Example:
            >>> model.lights
        """
        if "Lights" in self.epjs:
            return list(self.epjs["Lights"].keys())
        return []

    @property
    def zones(self):
        """
        Example:
            >>> model.zones
        """
        if "Zone" in self.epjs:
            return list(self.epjs["Zone"].keys())
        return []

    def _add(self, key: str, obj: Union[dict, str]):
        """Add an object to the epjs dictionary.

        Args:
            key: Key of the object to be added.
            obj: Object to be added.
        """
        if key in self.epjs:
            # merge
            self.epjs[key] = {**self.epjs[key], **obj}
        else:
            # add
            self.epjs[key] = obj

    def add_glazing_system(self, glazing_system: GlazingSystem):
        """Add glazing system to EnergyPlusModel's epjs dictionary.

        Args:
            glazing_system: GlazingSystem object

        Example:
            >>> model.add_glazing_system(glazing_system1)
        """
        name = glazing_system.name
        if (
            glazing_system.solar_results is None
            and glazing_system.photopic_results is None
        ):
            glazing_system.compute_solar_photopic_results()
            solar_results = glazing_system.solar_results
            photopic_results = glazing_system.photopic_results
        else:
            solar_results = glazing_system.solar_results
            photopic_results = glazing_system.photopic_results

        # Initialize Construction:ComplexFenestrationState dictionary with system and outer layer names

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
                "thickness": gap[-1],
            }
            window_material_gas[_gas_name] = {
                "gas_type": gap[0][0].name.capitalize(),
                "thickness": gap[-1],
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

    def add_lighting(self, zone: str, replace: bool = False):
        """Add lighting object to EnergyPlusModel's epjs dictionary.

        Args:
            zone: Zone name to add lighting to.
            replace: If True, replace existing lighting object in zone.

        Raises:
            ValueError: If zone not found in model.
            ValueError: If lighting already exists in zone and replace is False.

        Example:
            >>> model.add_lighting("Zone1")
        """
        if zone in self.zones:
            pass
        else:
            raise ValueError(f"Zone = {zone} not found in model.")

        dict2 = copy.deepcopy(self.epjs["Lights"])

        if self.epjs["Lights"] is not None:
            for light in dict2:
                if (
                    self.epjs["Lights"][light][
                        "zone_or_zonelist_or_space_or_spacelist_name"
                    ]
                    == zone
                ):
                    if replace:
                        del self.epjs["Lights"][light]
                    else:
                        raise ValueError(
                            f"Lighting already exists in zone = {zone}. "
                            "To replace, set replace=True."
                        )

        # Add lighting schedule type limit to epjs dictionary
        schedule_type_limit = {
            "on_off": {
                "lower_limit_value": 0,
                "upper_limit_value": 1,
                "numeric_type": "Discrete",
                "unit_type": "Availability",
            }
        }
        self._add("ScheduleTypeLimits", schedule_type_limit)

        # Add lighting schedule to epjs dictionary
        lighting_schedule = {
            "constant_off": {
                "schedule_type_limits_name": "on_off",
                "hourly_value": 0,
            }
        }
        self._add("Schedule:Constant", lighting_schedule)

        # Add lighting to epjs dictionary
        lights = {
            f"Light_{zone}": {
                "design_level_calculation_method": "LightingLevel",
                "fraction_radiant": 0,
                "fraction_replaceable": 1,
                "fraction_visible": 1,
                "lighting_level": 0,
                "return_air_fraction": 0,
                "schedule_name": "constant_off",
                "zone_or_zonelist_or_space_or_spacelist_name": zone,
            }
        }
        self._add("Lights", lights)

    def add_output(
        self, output_type: str, output_name: str, reporting_frequency: str = "Timestep"
    ):
        """Add an output variable or meter to the epjs dictionary.

        Args:
            output_type: Type of the output. "variable" or "meter".
            output_name: Name of the output variable or meter.
            reporting_frequency: Reporting frequency of the output variable or meter.

        Raises:
            ValueError: If output_type is not "variable" or "meter".

        Example:
            >>> model.add_output("Zone Mean Air Temperature", "variable")
            >>> model.add_output("Cooling:Electricity", "meter")
        """

        if output_type == "variable":
            self._add_output_variable(output_name, reporting_frequency)
        elif output_type == "meter":
            self._add_output_meter(output_name, reporting_frequency)
        else:
            raise ValueError("output_type must be 'variable' or 'meter'.")

    def _add_output_variable(self, output_name: str, reporting_frequency):
        """Add an output variable to the epjs dictionary.

        Args:
            output_name: Name of the output variable.
            reporting_frequency: Reporting frequency of the output variable.
        """
        i = 1
        if "Output:Variable" not in self.epjs:
            self.epjs["Output:Variable"] = {}
        for output in self.epjs["Output:Variable"].values():
            i += 1
            if output["variable_name"] == output_name:
                break
        else:
            self.epjs["Output:Variable"][f"Output:Variable {i}"] = {
                "key_value": "*",
                "reporting_frequency": reporting_frequency,
                "variable_name": output_name,
            }

    def _add_output_meter(self, output_name: str, reporting_frequency):
        """Add an output meter to the epjs dictionary.

        Args:
            output_name: Name of the output meter.
            reporting_frequency: Reporting frequency of the output meter.
        """
        i = 1
        if "Output:Meter" not in self.epjs:
            self.epjs["Output:Meter"] = {}
        for output in self.epjs["Output:Meter"].values():
            i += 1
            if output["key_name"] == output_name:
                break
        else:
            self.epjs["Output:Meter"][f"Output:Meter {i}"] = {
                "key_name": output_name,
                "reporting_frequency": reporting_frequency,
            }


def ep_datetime_parser(inp: str):
    """Parse date and time from EnergyPlus output.

    Args:
        inp: Date and time string from EnergyPlus output.
    """
    date, time = inp.strip().split()
    month, day = [int(i) for i in date.split("/")]
    hr, mi, sc = [int(i) for i in time.split(":")]
    if hr == 24 and mi == 0 and sc == 0:
        return datetime(1900, month, day, 0, mi, sc) + timedelta(days=1)
    else:
        return datetime(1900, month, day, hr, mi, sc)


class EnergyPlusSetup:
    """EnergyPlus Simulation Setup.

    Attributes:
        api: EnergyPlusAPI object
        epjs: EnergyPlusJSON object
        state: EnergyPlusState object
        handles: Handles object
        wea_meta: WeaMetaData object
    """

    def __init__(self, epmodel: EnergyPlusModel, weather_file: Optional[str] = None):
        """Class for setting up and running EnergyPlus simulations.

        Args:
            epmodel: EnergyPlusModel object

        Example:
            >>> epsetup = EnergyPlusSetup(epmodel, epw="USA_CA_Oakland.Intl.AP.724930_TMY3.epw")
        """
        self.api = epmodel.api
        self.epw = weather_file
        self.epjs = epmodel.epjs
        self.state = self.api.state_manager.new_state()
        self.variable_handles = {}
        self.actuator_handles = {}
        self.construction_handles = {}

        loc = list(self.epjs["Site:Location"].values())[0]
        self.wea_meta = sky.WeaMetaData(
            city=list(self.epjs["Site:Location"].keys())[0],
            country="",
            elevation=loc["elevation"],
            latitude=loc["latitude"],
            longitude=0 - loc["longitude"],
            timezone=(0 - loc["time_zone"]) * 15,
        )

        self.api.runtime.callback_begin_new_environment(self.state, self._get_handles())
        self.actuators = None
        self._get_list_of_actuators()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.api.state_manager.delete_state(self.state)

    def _actuator_func(self, state):
        actuators_list = []
        if self.actuators is None:
            list = self.api.api.listAllAPIDataCSV(state).decode("utf-8")
            for line in list.split("\n"):
                if line.startswith("Actuator"):
                    line = line.replace(";", "")
                    actuators_list.append(line.split(",", 1)[1])
            self.actuators = actuators_list
        else:
            self.api.api.stopSimulation(state)

    def _get_list_of_actuators(self):
        with open("epmodel.json", "w") as wtr:
            json.dump(self.epjs, wtr)

        actuator_state = self.api.state_manager.new_state()
        self.api.runtime.set_console_output_status(actuator_state, False)
        method = getattr(
            self.api.runtime, "callback_begin_system_timestep_before_predictor"
        )
        method(actuator_state, self._actuator_func)

        if self.epw is not None:
            self.api.runtime.run_energyplus(
                actuator_state, ["-p", "actuator", "-w", self.epw, "epmodel.json"]
            )
        elif "SizingPeriod:DesignDay" in self.epjs:
            self.api.runtime.run_energyplus(actuator_state, ["-D", "epmodel.json"])
        else:
            raise ValueError(
                "Specify weather file in EnergyPlusSetup "
                "or model design day in EnergyPlusModel."
            )
        self.api.state_manager.delete_state(actuator_state)

    def actuate(self, component_type: str, name: str, key: str, value: float):
        """Set or update the operating value of an actuator in the EnergyPlus model.

        If actuator has not been requested previously, it will be requested.
        Set the actuator value to the value specified.

        Args:
            component_type: The actuator category, e.g. "Weather Data"
            name: The name of the actuator to retrieve, e.g. "Outdoor Dew Point"
            key: The instance of the variable to retrieve, e.g. "Environment"
            value: The value to set the actuator to

        Raises:
            ValueError: If the actuator is not found

        Example:
            >>> epsetup.actuate("Weather Data", "Outdoor Dew Point", "Environment", 10)
        """
        if key not in self.actuator_handles:  # check if key exists in actuator handles
            self.actuator_handles[key] = {}
        if (
            name not in self.actuator_handles[key]
        ):  # check if name exists in actuator handles
            handle = self.api.exchange.get_actuator_handle(
                self.state, component_type, name, key
            )
            if handle == -1:
                del self.actuator_handles[key]
                raise ValueError(
                    "Actuator is not found: "
                    f"component_type = {component_type}, name = {name}, key = {key}."
                )
            self.actuator_handles[key][name] = handle

        # set actuator value
        self.api.exchange.set_actuator_value(
            self.state, self.actuator_handles[key][name], value
        )

    def get_variable_value(self, name: str, key: str) -> float:
        """Get the value of a variable in the EnergyPlus model during runtime.

        Args:
            name: The name of the variable to retrieve, e.g. "Outdoor Dew Point"
            key: The instance of the variable to retrieve, e.g. "Environment"

        Returns:
            The value of the variable

        Raises:
            ValueError: If the variable is not found

        Example:
            >>> epsetup.get_variable_value("Outdoor Dew Point", "Environment")
        """
        return self.api.exchange.get_variable_value(
            self.state, self.variable_handles[key][name]
        )

    def request_variable(self, name: str, key: str):
        """Request a variable from the EnergyPlus model during runtime.

        Args:
            name: The name of the variable to retrieve, e.g. "Outdoor Dew Point"
            key: The instance of the variable to retrieve, e.g. "Environment"

        Example:
            >>> epsetup.request_variable("Outdoor Dew Point", "Environment")
        """
        if key not in self.variable_handles:
            self.variable_handles[key] = {}
        if name in self.variable_handles[key]:
            pass
        else:
            self.api.exchange.request_variable(self.state, name, key)
            self.variable_handles[key][name] = None

    def _get_handles(self):
        def callback_function(state):
            for key in self.variable_handles:
                try:
                    for name in self.variable_handles[key]:
                        handle = self.api.exchange.get_variable_handle(state, name, key)
                        if handle == -1:
                            raise ValueError(
                                "Variable handle not found: "
                                f"name = {name}, key = {key}"
                            )
                        self.variable_handles[key][name] = handle
                except TypeError:
                    print("No variables requested for", self.variable_handles, key)

            if "Construction:ComplexFenestrationState" in self.epjs:
                for cfs in self.epjs["Construction:ComplexFenestrationState"]:
                    handle = self.api.api.getConstructionHandle(state, cfs.encode())
                    if handle == -1:
                        raise ValueError(
                            "Construction handle not found: " f"Construction = {cfs}"
                        )
                    self.construction_handles[cfs] = handle

        return callback_function

    def get_datetime(self) -> datetime:
        """Get the current date and time from EnergyPlus

        Returns:
            datetime object
        """
        year = self.api.exchange.year(self.state)
        month = self.api.exchange.month(self.state)
        day = self.api.exchange.day_of_month(self.state)
        hour = self.api.exchange.hour(self.state)
        minute = self.api.exchange.minutes(self.state)

        date = datetime(year, month, day)

        if minute == 60:
            minute = 0
            hour += 1

        if hour == 24:
            hour = 0
            date += timedelta(days=1)

        dt = date + timedelta(hours=hour, minutes=minute)

        return dt

    def run(
        self,
        output_directory: Optional[str] = "./",
        output_prefix: Optional[str] = "eplus",
        output_suffix: Optional[str] = "L",
        silent: bool = False,
        annual: bool = False,
        design_day: bool = False,
    ):
        """Run EnergyPlus simulation.

        Args:
            output_directory: Output directory path. (default: current directory)
            output_prefix: Prefix for output files. (default: eplus)
            output_suffix: Suffix style for output files. (default: L)
                L: Legacy (e.g., eplustbl.csv)
                C: Capital (e.g., eplusTable.csv)
                D: Dash (e.g., eplus-table.csv)
            silent: If True, do not print EnergyPlus output to console. (default: False)
            annual: If True, force run annual simulation. (default: False)
            design_day: If True, force run design-day-only simulation. (default: False)

        Example:
            >>> epsetup.run(output_prefix="test1", silent=True)
        """
        opt = ["-d", output_directory, "-p", output_prefix, "-s", output_suffix]

        if self.epw is not None:
            opt.extend(["-w", self.epw])
        elif design_day:
            if "SizingPeriod:DesignDay" in self.epjs:
                opt.append("-D")
            else:
                raise ValueError(
                    "Design day simulation requested, "
                    "but no design day found in EnergyPlus model."
                )
        else:
            raise ValueError(
                "Specify weather file in EnergyPlusSetup or "
                "run with design_day = True for design-day-only simulation."
            )

        if annual:
            if self.epw is not None:
                opt.append("-a")
            else:
                raise ValueError(
                    "Annual simulation requested, but no weather file found."
                )

        if "OutputControl:Files" not in self.epjs:
            self.epjs["OutputControl:Files"] = {
                "OutputControl:Files 1": {"output_csv": "Yes"}
            }

        with open(f"{output_prefix}.json", "w") as wtr:
            json.dump(self.epjs, wtr)

        self.api.runtime.set_console_output_status(self.state, not silent)
        self.api.runtime.run_energyplus(self.state, [*opt, f"{output_prefix}.json"])

    def set_callback(self, method_name: str, func: Callable):
        """Set callback function for EnergyPlus runtime API.

        Args:
            method_name: Name of the method to set callback for.
            func: Callback function.

        Raises:
            AttributeError: If method_name is not found in EnergyPlus runtime API.

        Example:
            >>> epsetup.set_callback("callback_begin_system_timestep_before_predictor", func)
        """
        try:
            method = getattr(self.api.runtime, method_name)
        except AttributeError:
            raise AttributeError(
                f"Method {method_name} not found in EnergyPlus runtime API."
            )
        # method(self.state, partial(func, self))
        method(self.state, func)
