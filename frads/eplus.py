"""
Class and functions for accessing EnergyPlus Python API
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import List, Optional, Callable, Union
import inspect
import ast

from epmodel import epmodel as epm
import epmodel
from frads import sky
from frads.window import GlazingSystem
import copy
from pyenergyplus.api import EnergyPlusAPI


class EnergyPlusModel(epmodel.EnergyPlusModel):
    """EnergyPlus Model object

    Attributes:
        cfs: list of complex fenestration states
        windows: list of windows
        walls_window: list of walls with windows
        floors: list of floors
        lighting_zone: list of lighting zones
        zones: list of zones
    """

    @property
    def window_walls(self) -> List[str]:
        """
        Example:
            >>> model.window_walls
        """
        if self.fenestration_surface_detailed is None:
            return []
        wndo_walls = {
            srf.building_surface_name
            for srf in self.fenestration_surface_detailed.values()
        }
        return list(wndo_walls)

    @property
    def floors(self):
        """
        Examples:
            >>> model.floors
        """
        floors = []
        if self.building_surface_detailed is None:
            return []
        for k, v in self.building_surface_detailed.items():
            if v.surface_type == epm.SurfaceType.floor:
                floors.append(k)
        return floors

    def add_glazing_system(self, glzsys: GlazingSystem):
        """Add glazing system to EnergyPlusModel's epjs dictionary.

        Args:
            glzsys: GlazingSystem object

        Raises:
            ValueError: If solar and photopic results are not computed.

        Example:
            >>> model = load_energyplus_model(Path("model.idf"))
            >>> model.add_glazing_system(glazing_system1)
        """
        if glzsys.solar_results is None or glzsys.photopic_results is None:
            glzsys.compute_solar_photopic_results()
        if glzsys.solar_results is None or glzsys.photopic_results is None:
            raise ValueError("Solar and photopic results not computed.")

        name = glzsys.name
        gap_inputs = []
        for i, gap in enumerate(glzsys.gaps):
            gap_inputs.append(
                epmodel.ConstructionComplexFenestrationStateGapInput(
                    gas=getattr(epm.GasType, gap[0][0].name.lower()), thickness=gap[-1]
                )
            )
        layer_inputs: List[epmodel.ConstructionComplexFenestrationStateLayerInput] = []
        for i, layer in enumerate(glzsys.layers):
            layer_inputs.append(
                epmodel.ConstructionComplexFenestrationStateLayerInput(
                    name=f"{glzsys.name}_layer_{i}",
                    product_type=layer.product_type,
                    thickness=layer.thickness,
                    conductivity=layer.conductivity,
                    emissivity_front=layer.emissivity_front,
                    emissivity_back=layer.emissivity_back,
                    infrared_transmittance=layer.ir_transmittance,
                    directional_absorptance_front=glzsys.solar_results.layer_results[
                        i
                    ].front.absorptance.angular_total,
                    directional_absorptance_back=glzsys.solar_results.layer_results[
                        i
                    ].back.absorptance.angular_total,
                )
            )
        input = epmodel.ConstructionComplexFenestrationStateInput(
            gaps=gap_inputs,
            layers=layer_inputs,
            solar_reflectance_back=glzsys.solar_results.system_results.back.reflectance.matrix,
            solar_transmittance_back=glzsys.solar_results.system_results.front.transmittance.matrix,
            visible_transmittance_back=glzsys.photopic_results.system_results.back.transmittance.matrix,
            visible_transmittance_front=glzsys.photopic_results.system_results.front.transmittance.matrix,
        )
        self.add_construction_complex_fenestration_state(name, input)

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
        if self.zone is None:
            raise ValueError("Zone not found in model.")
        if zone not in self.zone:
            raise ValueError(f"{zone} not found in model.")
        if self.lights is None:
            raise ValueError("Lights not found in model.")
        dict2 = copy.deepcopy(self.lights)

        if self.lights is not None:
            for light in dict2.values():
                if light.zone_or_zonelist_or_space_or_spacelist_name == zone:
                    if replace:
                        del light
                    else:
                        raise ValueError(
                            f"Lighting already exists in zone = {zone}. "
                            "To replace, set replace=True."
                        )

        # Add lighting schedule type limit to epjs dictionary
        self.add(
            "schedule_type_limits",
            "on_off",
            epm.ScheduleTypeLimits(
                lower_limit_value=0,
                upper_limit_value=1,
                numeric_type=epm.NumericType.discrete,
                unit_type=epm.UnitType.availability,
            ),
        )

        # Add lighting schedule to epjs dictionary
        self.add(
            "schedule_constant",
            "constant_off",
            epm.ScheduleConstant(
                schedule_type_limits_name="on_off",
                hourly_value=0,
            ),
        )

        # Add lighting to epjs dictionary
        self.add(
            "lights",
            f"Light_{zone}",
            epm.Lights(
                design_level_calculation_method=epm.DesignLevelCalculationMethod.lighting_level,
                fraction_radiant=0,
                fraction_replaceable=1,
                fraction_visible=1,
                lighting_level=0,
                return_air_fraction=0,
                schedule_name="constant_off",
                zone_or_zonelist_or_space_or_spacelist_name=zone,
            ),
        )

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
        if self.output_variable is None:
            self.output_variable = {}
        for output in self.output_variable.values():
            i += 1
            if output.variable_name == output_name:
                break
        else:
            self.output_variable[f"Output:Variable {i}"] = epm.OutputVariable(
                key_value="*",
                reporting_frequency=reporting_frequency,
                variable_name=output_name,
            )

    def _add_output_meter(self, output_name: str, reporting_frequency):
        """Add an output meter to the epjs dictionary.

        Args:
            output_name: Name of the output meter.
            reporting_frequency: Reporting frequency of the output meter.
        """
        i = 1
        if self.output_meter is None:
            self.output_meter = {}
        for output in self.output_meter.values():
            i += 1
            if output.key_name == output_name:
                break
        else:
            self.output_meter[f"Output:Meter {i}"] = epm.OutputMeter(
                key_name=output_name,
                reporting_frequency=reporting_frequency,
            )


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


class EnergyPlusResult:
    def __init__(self):
        ...


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
        self.model = epmodel
        self.api = EnergyPlusAPI()
        self.epw = weather_file
        self.state = self.api.state_manager.new_state()
        self.result = EnergyPlusResult()
        self.variable_handles = {}
        self.actuator_handles = {}
        self.construction_handles = {}

        if self.model.site_location is None:
            raise ValueError("Site location not found in EnergyPlus model.")

        city, location = next(iter(self.model.site_location.items()))
        self.wea_meta = sky.WeaMetaData(
            city=city,
            country="",
            elevation=location.elevation or 0,
            latitude=location.latitude or 0,
            longitude=-(location.longitude or 0),
            timezone=-(location.time_zone or 0) * 15,
        )

        self.api.runtime.callback_begin_new_environment(self.state, self._get_handles())
        self.actuators = []
        self._get_list_of_actuators()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.api.state_manager.delete_state(self.state)

    def _actuator_func(self, state):
        if len(self.actuators) == 0:
            api_data: List[str] = self.api.api.listAllAPIDataCSV(state).decode("utf-8").splitlines()
            for line in api_data:
                if line.startswith("Actuator"):
                    line = line.replace(";", "")
                    self.actuators.append(line.split(",", 1)[1].split(","))
        else:
            self.api.api.stopSimulation(state)

    def _get_list_of_actuators(self):
        with open("epmodel.json", "w") as wtr:
            wtr.write(self.model.model_dump_json(by_alias=True, exclude_none=True))

        actuator_state = self.api.state_manager.new_state()
        self.api.runtime.set_console_output_status(actuator_state, False)
        self.api.runtime.callback_begin_system_timestep_before_predictor(actuator_state, self._actuator_func)

        if self.epw is not None:
            self.api.runtime.run_energyplus(
                actuator_state, ["-p", "actuator", "-w", self.epw, "epmodel.json"]
            )
        elif self.model.sizing_period_design_day is not None:
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
            KeyError: If the key is not found
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

            if self.model.construction_complex_fenestration_state is not None:
                for cfs in self.model.construction_complex_fenestration_state:
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
    ) -> None:
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
            if self.model.sizing_period_design_day is not None:
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

        if self.model.output_control_files is None:
            self.model.output_control_files = {
                "OutputControl:Files 1": epm.OutputControlFiles(
                    output_csv=epm.EPBoolean.yes
                )
            }

        if self.model.output_control_timestamp is None:
            self.model.output_control_timestamp = {
                "OutputControl:Timestamp 1": epm.OutputControlTimestamp(
                    iso8601_format=epm.EPBoolean.yes,
                    timestamp_at_beginning_of_interval=epm.EPBoolean.yes,
                )
            }

        with open(f"{output_prefix}.json", "w") as wtr:
            wtr.write(self.model.model_dump_json(by_alias=True, exclude_none=True))

        self.api.runtime.set_console_output_status(self.state, not silent)
        self.api.runtime.run_energyplus(self.state, [*opt, f"{output_prefix}.json"])

        # TODO: Setup temp directory for EnergyPlus output
        # with TemporaryDirectory() as temp_dir:
        #     with open(os.path.join(temp_dir, "input.json"), "w") as wtr:
        #         wtr.write(self.model.model_dump_json(by_alias=True, exclude_none=True))
        #     opt.extend(["-d", temp_dir, "input.json"])
        #     self.api.runtime.set_console_output_status(self.state, not silent)
        #     self.api.runtime.run_energyplus(self.state, opt)
        #     # load everything except input.json and sqlite.err
        #     for ofile in os.listdir(temp_dir):
        #         if ofile not in ["input.json", "sqlite.err"]:
        #             with open(os.path.join(temp_dir, ofile)) as f:
        #                 setattr(self.result, ofile, f.read())
        # return self.result

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

        self._request_variable_from_callback(func)

        # method(self.state, partial(func, self))
        method(self.state, func)

    def _analyze_callback(self, func: Callable) -> None:
        """Request variables from callback function.

        Args:
            func: Callback function.

        Example:
            >>> epsetup._request_variable_from_callback(func)
        """
        source_code = inspect.getsource(func)
        tree = ast.parse(source_code)
        key_value_pairs = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, 'attr'):
                if node.func.attr == 'get_variable_value':
                    if len(node.args) == 2:
                        key_value_dict = {
                            "name": ast.literal_eval(node.args[0]),
                            "key": ast.literal_eval(node.args[1]),
                        }
                    elif len(node.keywords) == 2:
                        key_value_dict = {
                            node.keywords[0].arg: node.keywords[0].value.value,
                            node.keywords[1].arg: node.keywords[1].value.value,
                        }
                    else:
                        raise ValueError(f"Invalid number of arguments in {func}.")
                    key_value_pairs.append(key_value_dict)

                if node.func.attr == 'actuate':
                    if len(node.args) == 4:
                        key_value = [ast.literal_eval(node.args[i]) for i in range(4)]
                    elif len(node.keywords) == 4:
                        key_value_dict = {node.keywords[i].arg: node.keywords[i].value.value for i in range(4)}
                        key_value= [key_value_dict['component_type'], key_value_dict['name'], key_value_dict['key'], key_value_dict['value']]
                    else:
                        raise ValueError(f"Invalid number of arguments in {func}.")
                    if key_value not in self.actuators:
                        raise ValueError(f"Actuator {key_value} not found in model.")

        for key_value_dict in key_value_pairs:
            self.request_variable(**key_value_dict)




def load_idf(fpath: Union[str, Path]) -> dict:
    """Load IDF file as JSON object.

    Use EnergyPlus --convert-only option to convert IDF file to epJSON file.

    Args:
        fpath: Path to IDF file.

    Returns:
        JSON object.

    Raises:
        ValueError: If file is not an IDF file.
        FileNotFoundError: If IDF file not found.
        FileNotFoundError: If converted epJSON file not found.

    Example:
        >>> json_data = load_idf_as_json(Path("model.idf"))
    """
    fpath = Path(fpath) if isinstance(fpath, str) else fpath
    if fpath.suffix != ".idf":
        raise ValueError(f"File {fpath} is not an IDF file.")
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} not found.")
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    api.runtime.set_console_output_status(state, False)
    api.runtime.run_energyplus(state, ["--convert-only", str(fpath)])
    api.state_manager.delete_state(state)
    epjson_path = Path(fpath.with_suffix(".epJSON").name)
    if not epjson_path.exists():
        raise FileNotFoundError(f"Converted {str(epjson_path)} not found.")
    with open(epjson_path) as f:
        json_data = json.load(f)
    epjson_path.unlink()
    return json_data


def load_energyplus_model(fpath: Union[str, Path]) -> EnergyPlusModel:
    """Load EnergyPlus model from JSON file.

    Args:
        fpath: Path to JSON file.

    Returns:
        EnergyPlusModel object.

    Raises:
        ValueError: If file is not an IDF or epJSON file.

    Example:
        >>> model = load_energyplus_model(Path("model.json"))
    """
    fpath = Path(fpath) if isinstance(fpath, str) else fpath
    if fpath.suffix == ".idf":
        json_data = load_idf(fpath)
    elif fpath.suffix == ".epJSON":
        with open(fpath) as f:
            json_data = json.load(f)
    else:
        raise ValueError(f"File {fpath} is not an IDF or epJSON file.")
    return EnergyPlusModel.model_validate(json_data)
