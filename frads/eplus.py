"""
Class and functions for accessing EnergyPlus Python API
"""

import ast
import datetime
import inspect
import json
import tempfile
import textwrap
import os
from pathlib import Path
from typing import Callable

from epmodel import epmodel as epm
import frads as fr
from frads.eplus_model import EnergyPlusModel
from .window import GlazingSystem
from frads.methods import MatrixConfig
import numpy as np
from pyenergyplus.api import EnergyPlusAPI


def ep_datetime_parser(inp: str) -> datetime.datetime:
    """Parse date and time from EnergyPlus output format.

    EnergyPlus outputs dates and times in "MM/DD HH:MM:SS" format. This function
    handles the special case where EnergyPlus uses "24:00:00" to represent midnight
    of the next day.

    Args:
        inp: Date and time string from EnergyPlus output (e.g., "01/15 14:30:00").

    Returns:
        A datetime object representing the parsed date and time.

    Examples:
        >>> ep_datetime_parser("01/15 14:30:00")
        datetime.datetime(1900, 1, 15, 14, 30)
        >>> ep_datetime_parser("12/31 24:00:00")  # Midnight of next day
        datetime.datetime(1901, 1, 1, 0, 0)
    """
    date, time = inp.strip().split()
    month, day = [int(i) for i in date.split("/")]
    hr, mi, sc = [int(i) for i in time.split(":")]
    if hr == 24 and mi == 0 and sc == 0:
        return datetime.datetime(1900, month, day, 0, mi, sc) + datetime.timedelta(
            days=1
        )
    else:
        return datetime.datetime(1900, month, day, hr, mi, sc)


class EnergyPlusResult:
    def __init__(self): ...


class EnergyPlusSetup:
    """EnergyPlus Simulation Setup.

    Attributes:
        api: EnergyPlusAPI object
        epw: Weather file path
        actuator_handles: Actuator Handles
        variable_handles: Variable handles
        construction_handles: Construction Handles
        actuators: List of actuators available
        model: EnergyPlusModel object
        state: EnergyPlusState object
        handles: Handles object
    """

    def __init__(
        self,
        epmodel: EnergyPlusModel,
        weather_file: None | str = None,
        enable_radiance: bool = False,
        nproc: int = 1,
        initialize_radiance: bool = True,
    ):
        """Class for setting up and running EnergyPlus simulations.

        Args:
            epmodel: EnergyPlusModel object
            weather_file: Weather file path. (default: None)
            enable_radiance: If True, enable Radiance for Three-Phase Method. (default: False)

        Examples:
            >>> epsetup = EnergyPlusSetup(epmodel, weather_file="USA_CA_Oakland.Intl.AP.724930_TMY3.epw")
        """
        self.model = epmodel

        if self.model.site_location is None:
            raise ValueError("Site location not found in EnergyPlus model.")

        self.rmodels = {}
        self.rconfigs = {}
        self.rworkflows = {}
        if enable_radiance:
            self.rmodels = fr.epmodel_to_radmodel(
                epmodel, epw_file=weather_file, add_views=True
            )
            self.rconfigs = {
                k: fr.WorkflowConfig.from_dict(v) for k, v in self.rmodels.items()
            }
            # Default to Three-Phase Method
            self.rworkflows = {
                k: fr.ThreePhaseMethod(v) for k, v in self.rconfigs.items()
            }
            if initialize_radiance:
                self.initialize_radiance(nproc=nproc)
        self.api = EnergyPlusAPI()
        self.epw = weather_file
        self.state = self.api.state_manager.new_state()
        self.result = EnergyPlusResult()
        self.variable_handles = {}
        self.actuator_handles = {}
        self.construction_handles = {}
        self.construction_names = {}
        self.enable_radiance = enable_radiance
        self.api.runtime.callback_begin_new_environment(self.state, self._get_handles())
        self.actuators = []
        self._get_list_of_actuators()

    def initialize_radiance(self, zones: None | list[str] = None, nproc: int = 1):
        """Initialize Radiance for Three-Phase Method.

        Args:
            zones: List of zones to initialize. If None, initialize all zones.
            nproc: Number of processors to use for generating matrices.
        """
        if zones is None:
            zones = list(self.rworkflows.keys())
        for zone in zones:
            self.rworkflows[zone].config.settings.save_matrices = True
            self.rworkflows[zone].config.settings.num_processors = nproc
            self.rworkflows[zone].generate_matrices(view_matrices=False)

    def close(self):
        self.api.state_manager.delete_state(self.state)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _actuator_func(self, state):
        if len(self.actuators) == 0:
            api_data: list[str] = (
                self.api.api.listAllAPIDataCSV(state).decode("utf-8").splitlines()
            )
            for line in api_data:
                if line.startswith("Actuator"):
                    line = line.replace(";", "")
                    self.actuators.append(line.split(",", 1)[1].split(","))
        else:
            self.api.api.stopSimulation(state)

    def _get_list_of_actuators(self):
        actuator_state = self.api.state_manager.new_state()
        self.api.runtime.set_console_output_status(actuator_state, False)
        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
            actuator_state, self._actuator_func
        )

        tmpdir = tempfile.mkdtemp()
        inp = os.path.join(tmpdir, "in.json")
        with open(inp, "w") as wtr:
            wtr.write(self.model.model_dump_json(by_alias=True, exclude_none=True))

        if self.epw is not None:
            self.api.runtime.run_energyplus(
                actuator_state, ["-p", "actuator", "-d", tmpdir, "-w", self.epw, inp]
            )
        elif self.model.sizing_period_design_day is not None:
            self.api.runtime.run_energyplus(actuator_state, ["-D", "-d", tmpdir, inp])
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

        Examples:
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

    def actuate_cooling_setpoint(self, zone: str, value: float) -> None:
        """Set cooling setpoint for a zone.

        Args:
            zone: The name of the zone to set the cooling setpoint for.
            value: The value to set the cooling setpoint to.

        Examples:
            >>> epsetup.actuate_cooling_setpoint("zone1", 24)
        """
        self.actuate(
            component_type="Zone Temperature Control",
            name="Cooling Setpoint",
            key=zone,
            value=value,
        )

    def actuate_heating_setpoint(self, zone: str, value: float):
        """Set heating setpoint for a zone.

        Args:
            zone: The name of the zone to set the heating setpoint for.
            value: The value to set the heating setpoint to.

        Example:
            epsetup.actuate_cooling_setpoint("zone1", 20)
        """
        self.actuate(
            component_type="Zone Temperature Control",
            name="Heating Setpoint",
            key=zone,
            value=value,
        )

    def actuate_lighting_power(self, light: str, value: float):
        """Set lighting power for a zone.

        Args:
            light: The name of the lighting object to set the lighting power for.
            value: The value to set the lighting power to.

        Examples:
            >>> epsetup.actuate_lighting_power("zone1", 1000)
        """
        self.actuate(
            component_type="Lights",
            name="Electricity Rate",
            key=light,
            value=value,
        )

    def actuate_cfs_state(self, window: str, cfs_state: str):
        """Set construction state for a surface.

        Args:
            window: The name of the surface to set the CFS state for.
            cfs_state: The name of the complex fenestration system (CFS) state to set the surface to.

        Examples:
            >>> epsetup.actuate_cfs_state("window1", "cfs1")
        """
        self.actuate(
            component_type="Surface",
            name="Construction State",
            key=window,
            value=self.construction_handles[cfs_state],
        )

    def get_variable_value(self, name: str, key: str) -> float:
        """Get the value of a variable in the EnergyPlus model during runtime.
        The variable must be requested before it can be retrieved.
        If this method is called in a callback function, the variable will be requested automatically.
        So avoid having other methods called get_variable_value in the callback function.

        Args:
            name: The name of the variable to retrieve, e.g. "Outdoor Dew Point"
            key: The instance of the variable to retrieve, e.g. "Environment"

        Returns:
            The value of the variable

        Raises:
            KeyError: If the key is not found
            ValueError: If the variable is not found

        Examples:
            >>> epsetup.get_variable_value("Outdoor Dew Point", "Environment")
        """
        return self.api.exchange.get_variable_value(
            self.state, self.variable_handles[key][name]
        )

    def get_cfs_state(self, window: str) -> str:
        """Return the current complex fenestration state with input window name

        Args:
            window: name of the window

        Returns:
            name of the cfs state
        """

        cfs_handle = self.api.exchange.get_actuator_value(
            self.state, self.actuator_handles[window]["Construction State"]
        )

        cfs_name = self.construction_names[cfs_handle]
        return cfs_name

    def request_variable(self, name: str, key: str) -> None:
        """Request a variable from the EnergyPlus model for access during runtime.

        Args:
            name: The name of the variable to retrieve, e.g. "Outdoor Dew Point"
            key: The instance of the variable to retrieve, e.g. "Environment"

        Returns:
            None

        Examples:
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
                                f"Variable handle not found: name = {name}, key = {key}"
                            )
                        self.variable_handles[key][name] = handle
                except TypeError:
                    print("No variables requested for", self.variable_handles, key)

            if self.model.construction_complex_fenestration_state is not None:
                for cfs in self.model.construction_complex_fenestration_state:
                    handle = self.api.api.getConstructionHandle(state, cfs.encode())
                    if handle == -1:
                        raise ValueError(
                            f"Construction handle not found: Construction = {cfs}"
                        )
                    self.construction_handles[cfs] = handle
                    self.construction_names[handle] = cfs
                for wname, window in self.model.fenestration_surface_detailed.items():
                    if (
                        window.construction_name
                        in self.model.construction_complex_fenestration_state
                    ):
                        self.actuate_cfs_state(
                            wname,
                            window.construction_name,
                        )

        return callback_function

    def get_datetime(self) -> datetime.datetime:
        """Get the current date and time from EnergyPlus
        Run time datatime format with iso_8601_format = yes.
        hour 0-23, minute 10 - 60
        v23.2.0

        Returns:
            datetime object
        """
        year = self.api.exchange.year(self.state)
        month = self.api.exchange.month(self.state)
        day = self.api.exchange.day_of_month(self.state)
        hour = self.api.exchange.hour(self.state)
        minute = self.api.exchange.minutes(self.state)

        _date = datetime.date(year, month, day)

        if minute == 60:
            minute = 0
            hour += 1
        if hour == 24:
            hour = 0
            _date += datetime.timedelta(days=1)
        _time = datetime.time(hour, minute)

        return datetime.datetime.combine(_date, _time)

    def run(
        self,
        output_directory: None | str = "./",
        output_prefix: None | str = "eplus",
        output_suffix: None | str = "L",
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

        Examples:
            >>> epsetup.run(output_prefix="test1", silent=True)
        """
        opt = ["-d", output_directory, "-p", output_prefix, "-s", output_suffix]

        if self.epw is not None:
            opt.extend(["-w", self.epw])
        if design_day:
            if self.model.sizing_period_design_day is not None:
                opt.append("-D")
            else:
                raise ValueError(
                    "Design day simulation requested, "
                    "but no design day found in EnergyPlus model."
                )
        if self.epw is None and not design_day:
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

        self.model.output_control_timestamp = {
            "OutputControl:Timestamp 1": epm.OutputControlTimestamp(
                iso_8601_format=epm.EPBoolean.no,
                timestamp_at_beginning_of_interval=epm.EPBoolean.yes,
            )
        }

        with open(f"{output_prefix}.json", "w") as wtr:
            wtr.write(
                self.model.model_dump_json(
                    by_alias=True, exclude_none=True, exclude_unset=True
                )
            )

        self.api.runtime.set_console_output_status(self.state, not silent)
        self.api.runtime.run_energyplus(self.state, [*opt, f"{output_prefix}.json"])

    def set_callback(self, method_name: str, func: Callable):
        """Set callback function for EnergyPlus runtime API.

        Args:
            method_name: Name of the method to set callback for.
            func: Callback function.

        Raises:
            AttributeError: If method_name is not found in EnergyPlus runtime API.

        Examples:
            >>> epsetup.set_callback("callback_begin_system_timestep_before_predictor", func)
        """
        try:
            method = getattr(self.api.runtime, method_name)
        except AttributeError:
            raise AttributeError(
                f"Method {method_name} not found in EnergyPlus runtime API."
            )

        self._analyze_callback(func)

        # method(self.state, partial(func, self))
        method(self.state, func)

    def _request_diffuse_horizontal_irradiance(self) -> None:
        self.request_variable(
            name="Site Diffuse Solar Radiation Rate per Area",
            key="Environment",
        )

    def _request_direct_normal_irradiance(self) -> None:
        self.request_variable(
            name="Site Direct Solar Radiation Rate per Area",
            key="Environment",
        )

    def _request_sky_diffuse_solar_radiation_luminous_efficacy(self) -> None:
        self.request_variable(
            name="Site Sky Diffuse Solar Radiation Luminous Efficacy",
            key="Environment",
        )

    def _request_beam_solar_radiation_luminous_efficacy(self) -> None:
        self.request_variable(
            name="Site Beam Solar Radiation Luminous Efficacy",
            key="Environment",
        )

    def _request_total_sky_cover(self) -> None:
        self.request_variable(
            name="Site Total Sky Cover",
            key="Environment",
        )

    def _request_variables_from_callback(self, callable_nodes: list[ast.Call]) -> None:
        for node in callable_nodes:
            key_value_dict = {}
            if node.func.attr == "get_variable_value":
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
                    raise ValueError(f"Invalid number of arguments in {node.func}.")
                self.request_variable(**key_value_dict)
            elif node.func.attr == "get_diffuse_horizontal_irradiance":
                self._request_diffuse_horizontal_irradiance()
            elif node.func.attr == "get_diffuse_horizontal_illuminance":
                self._request_diffuse_horizontal_irradiance()
                self._request_sky_diffuse_solar_radiation_luminous_efficacy()
            elif node.func.attr == "get_direct_normal_irradiance":
                self._request_direct_normal_irradiance()
            elif node.func.attr == "get_direct_normal_illuminance":
                self._request_direct_normal_irradiance()
                self._request_beam_solar_radiation_luminous_efficacy()
            elif node.func.attr == "get_total_sky_cover":
                self._request_total_sky_cover()
            elif node.func.attr == "calculate_mev":
                self._request_diffuse_horizontal_irradiance()
                self._request_direct_normal_irradiance()
                self._request_sky_diffuse_solar_radiation_luminous_efficacy()
                self._request_beam_solar_radiation_luminous_efficacy()
                self._request_total_sky_cover()
            elif node.func.attr in ("calculate_wpi", "calculate_edgps"):
                self._request_diffuse_horizontal_irradiance()
                self._request_direct_normal_irradiance()

    def _check_actuators_from_callback(self, callable_nodes: list[ast.Call]) -> None:
        def get_zone_from_pair_arg(node: ast.Call) -> str:
            if len(node.args) == 2:
                zone = ast.literal_eval(node.args[0])
            elif len(node.keywords) == 2:
                key_value_dict = {
                    node.keywords[i].arg: node.keywords[i].value.value for i in range(2)
                }
                zone = key_value_dict.get("zone", key_value_dict.get("surface", None))
            else:
                raise ValueError(f"Invalid number of arguments in {node}.")
            return zone

        for node in callable_nodes:
            key_value = None
            if node.func.attr == "actuate":
                if len(node.args) == 4:
                    key_value = [ast.literal_eval(node.args[i]) for i in range(3)]
                elif len(node.keywords) == 4:
                    key_value_dict = {
                        node.keywords[i].arg: node.keywords[i].value.value
                        for i in range(4)
                    }
                    key_value = [
                        key_value_dict["component_type"],
                        key_value_dict["name"],
                        key_value_dict["key"],
                    ]
                else:
                    raise ValueError(f"Invalid number of arguments in {node}.")
            elif node.func.attr == "actuate_cfs_state":
                zone = get_zone_from_pair_arg(node)
                key_value = ["Surface", "Construction State", zone]
            elif node.func.attr == "actuate_cooling_setpoint":
                zone = get_zone_from_pair_arg(node)
                key_value = [
                    "Zone Temperature Control",
                    "Cooling Setpoint",
                    zone,
                ]
            elif node.func.attr == "actuate_heating_setpoint":
                zone = get_zone_from_pair_arg(node)
                key_value = [
                    "Zone Temperature Control",
                    "Heating Setpoint",
                    zone,
                ]
            elif node.func.attr == "actuate_lighting_power":
                zone = get_zone_from_pair_arg(node)
                key_value = ["Lights", "Electricity Rate", zone]
            if key_value is None:
                continue
            if key_value not in self.actuators:
                raise ValueError(f"Actuator {key_value} not found in model.")

    def _analyze_callback(self, func: Callable) -> None:
        """Request variables from callback function.

        Args:
            func: Callback function.
        """
        source_code = inspect.getsource(func)
        dedented_source = textwrap.dedent(source_code)
        tree = ast.parse(dedented_source)
        callable_nodes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
                callable_nodes.append(node)
        self._request_variables_from_callback(callable_nodes)
        # self._check_actuators_from_callback(callable_nodes)

    def get_direct_normal_irradiance(self) -> float:
        """Get direct normal irradiance.

        Returns:
            Direct normal irradiance in W/m2.

        Examples:
            >>> epsetup.get_direct_normal_irradiance()
        """
        return self.get_variable_value(
            "Site Direct Solar Radiation Rate per Area", "Environment"
        )

    def get_direct_normal_illuminance(self) -> float:
        """Get direct normal illuminance.

        Returns:
            Direct normal illuminance in lux.

        Examples:
            >>> epsetup.get_direct_normal_illuminance()
        """
        return self.get_direct_normal_irradiance() * self.get_variable_value(
            "Site Beam Solar Radiation Luminous Efficacy", "Environment"
        )

    def get_diffuse_horizontal_irradiance(self) -> float:
        """Get diffuse horizontal irradiance.

        Returns:
            Diffuse horizontal irradiance in W/m2.

        Example:
            epsetup.get_diffuse_horizontal_irradiance()
        """
        return self.get_variable_value(
            "Site Diffuse Solar Radiation Rate per Area", "Environment"
        )

    def get_diffuse_horizontal_illuminance(self) -> float:
        """Get diffuse horizontal illuminance.

        Returns:
            Diffuse horizontal illuminance in lux.

        Example:
            epsetup.get_diffuse_horizontal_illuminance()
        """
        return self.get_diffuse_horizontal_irradiance() * self.get_variable_value(
            "Site Sky Diffuse Solar Radiation Luminous Efficacy", "Environment"
        )

    def get_total_sky_cover(self) -> float:
        """Get total sky cover.

        Returns:
            Total sky cover in fraction.

        Example:
            epsetup.get_total_sky_cover()
        """
        return self.get_variable_value("Site Total Sky Cover", "Environment") / 10.0

    def calculate_wpi(self, zone: str, cfs_name: dict[str, str]) -> np.ndarray:
        """Calculate workplane illuminance in a zone.

        Args:
            zone: Name of the zone.
            cfs_name: Name of the complex fenestration state.

        Returns:
            Workplane illuminance in lux.

        Raises:
            ValueError: If zone not found in model.

        Examples:
            >>> epsetup.calculate_wpi("Zone1", "CFS1")
        """
        if not self.enable_radiance:
            raise ValueError("Radiance is not enabled.")
        date_time = self.get_datetime()
        dni = self.get_direct_normal_irradiance()
        dhi = self.get_diffuse_horizontal_irradiance()
        sensor_name = next(iter(self.rconfigs[zone].model.sensors.keys()))
        return self.rworkflows[zone].calculate_sensor(
            sensor_name, cfs_name, date_time, dni, dhi
        )

    def calculate_mev(self, zone: str, cfs_name: dict[str, str]) -> float:
        """Calculate menalonpic illuminance.

        Args:
            zone: Name of the zone.
            cfs_name: Dictionary of windows and their complex fenestration state.

        Returns:
            Menalonpic vertical illuminance.

        Raises:
            KeyError: If zone not found in model.

        Examples:
            >>> epsetup.calculate_mev("Zone1", "CFS1")
        """
        date_time = self.get_datetime()
        dni = self.get_direct_normal_illuminance()
        dhi = self.get_diffuse_horizontal_illuminance()
        sky_cover = self.get_total_sky_cover()
        view_name = next(iter(self.rconfigs[zone].model.views.keys()))
        return self.rworkflows[zone].calculate_mev(
            view_name, cfs_name, date_time, dni, dhi, sky_cover
        )

    def calculate_edgps(
        self, zone: str, cfs_name: dict[str, str]
    ) -> tuple[float, float]:
        """Calculate enhanced simplified daylight glare probability in a zone.

        The view is positioned at the center of the zone by default, with direction
        facing the windows weighted by window area. Users can modify the view position
        and direction through rconfig before calling initialize_radiance().

        Args:
            zone: Name of the zone.
            cfs_name: Dictionary of windows and their complex fenestration state.

        Returns:
            Tuple of (enhanced simplified DGP, simplified DGP).

        Raises:
            KeyError: If zone not found in model.

        Examples:
            >>> edgp, sdgp = epsetup.calculate_edgps("Zone1", {"window1": "state1"})
        """
        date_time = self.get_datetime()
        dni = self.get_direct_normal_irradiance()
        dhi = self.get_diffuse_horizontal_irradiance()
        view_name = next(iter(self.rconfigs[zone].model.views.keys()))
        return self.rworkflows[zone].calculate_edgps(
            view_name, cfs_name, date_time, dni, dhi
        )

    def add_proxy_geometry(self, gs: GlazingSystem):
        for _, zone in self.rworkflows.items():
            for _, window in zone.config.model.windows.items():
                geom = fr.window.get_proxy_geometry(window.polygon, gs)
                window.proxy_geometry[gs.name] = b"\n".join(geom)

    def add_melanopic_bsdf(self, gs: GlazingSystem):
        for _, zone in self.rworkflows.items():
            zone.config.model.materials.matrices_mlnp[gs.name] = MatrixConfig(
                matrix_data=np.array(gs.melanopic_back_transmittance)
            )


def load_idf(fpath: str | Path) -> dict:
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

    Examples:
        >>> json_data = load_idf(Path("model.idf"))
    """
    fpath = Path(fpath) if isinstance(fpath, str) else fpath
    if fpath.suffix != ".idf":
        raise ValueError(f"File {fpath} is not an IDF file.")
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} not found.")
    fpath = fpath.absolute()
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        try:
            os.chdir(td)
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
        finally:
            os.chdir(original_dir)
    return json_data


def load_energyplus_model(fpath: str | Path) -> EnergyPlusModel:
    """Load EnergyPlus model from IDF or epJSON file.

    Supports loading EnergyPlus models from both IDF (Input Data File) and
    epJSON (EnergyPlus JSON) formats. IDF files are automatically converted
    to epJSON format during loading.

    Args:
        fpath: Path to the EnergyPlus model file (.idf or .epJSON/.json).

    Returns:
        EnergyPlusModel object containing the parsed building model data.

    Raises:
        ValueError: If file extension is not .idf, .epJSON, or .json.
        FileNotFoundError: If the specified file does not exist.

    Examples:
        >>> # Load from IDF file
        >>> model = load_energyplus_model("building.idf")
        >>> # Load from epJSON file
        >>> model = load_energyplus_model("building.epJSON")
        >>> print(f"Building has {len(model.zones)} zones")
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
