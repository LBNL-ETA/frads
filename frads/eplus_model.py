import copy

import epmodel
import epmodel.epmodel as epm

from frads.window import GlazingSystem


class EnergyPlusModel(epmodel.EnergyPlusModel):
    """EnergyPlus Model object

    Attributes:
        walls_window: list of walls with windows
        floors: list of floors
        lighting_zone: list of lighting zones
        zones: list of zones
    """

    @property
    def window_walls(self) -> list[str]:
        """Get list of walls with windows."""
        if self.fenestration_surface_detailed is None:
            return []
        wndo_walls = {
            srf.building_surface_name
            for srf in self.fenestration_surface_detailed.values()
        }
        return list(wndo_walls)

    @property
    def floors(self):
        """Get all of the floor names."""
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

        Examples:
            >>> model = load_energyplus_model(Path("model.idf"))
            >>> model.add_glazing_system(glazing_system1)
        """

        name = glzsys.name
        gap_inputs: list[epm.WindowMaterialGas | epm.WindowMaterialGasMixture] = []
        for i, gap in enumerate(glzsys.gaps):
            if len(gap.gas) == 1:
                gap_inputs.append(
                    epm.WindowMaterialGas(
                        gas_type=gap.gas[0].gas.capitalize(),
                        thickness=gap.thickness_m,
                    )
                )
            else:
                gas_dict = {
                    "thickness": gap.thickness_m,
                    "number_of_gases_in_mixture": len(gap.gas),
                }
                for idx, gas in enumerate(gap.gas):
                    gas_dict[f"gas_{idx + 1}_type"] = gas.gas.capitalize()
                    gas_dict[f"gas_{idx + 1}_fraction"] = gas.ratio
                gap_inputs.append(epm.WindowMaterialGasMixture.model_validate(gas_dict))

        layer_inputs: list[epmodel.ConstructionComplexFenestrationStateLayerInput] = []
        for i, layer in enumerate(glzsys.layers):
            layer_inputs.append(
                epmodel.ConstructionComplexFenestrationStateLayerInput(
                    name=f"{glzsys.name}_layer_{i}",
                    product_type=layer.product_type,
                    thickness=layer.thickness_m,
                    conductivity=layer.conductivity,
                    emissivity_front=layer.emissivity_front,
                    emissivity_back=layer.emissivity_back,
                    infrared_transmittance=layer.ir_transmittance,
                    directional_absorptance_front=glzsys.solar_front_absorptance[i],
                    directional_absorptance_back=glzsys.solar_back_absorptance[i],
                    top_opening_multiplier=layer.opening_multipliers.top,
                    bottom_opening_multiplier=layer.opening_multipliers.bottom,
                    left_side_opening_multiplier=layer.opening_multipliers.left,
                    right_side_opening_multiplier=layer.opening_multipliers.right,
                    front_opening_multiplier=layer.opening_multipliers.front,
                    slat_width=layer.slat_width_m,
                    slat_spacing=layer.slat_spacing_m,
                    slat_thickness=layer.slat_thickness_m,
                    slat_angle=layer.slat_angle_deg,
                    slat_conductivity=layer.slat_conductivity,
                    slat_curve=layer.slat_curve_m,
                )
            )
        input = epmodel.ConstructionComplexFenestrationStateInput(
            gaps=gap_inputs,
            layers=layer_inputs,
            solar_reflectance_back=glzsys.solar_back_reflectance,
            solar_transmittance_front=glzsys.solar_front_transmittance,
            visible_reflectance_back=glzsys.visible_back_reflectance,  # This is correct, there is typo in schema
            visible_transmittance_front=glzsys.visible_front_transmittance,
        )
        self.add_construction_complex_fenestration_state(name, input)

    def add_lighting(self, zone: str, lighting_level: float, replace: bool = False):
        """Add lighting object to EnergyPlusModel's epjs dictionary.

        Args:
            zone: Zone name to add lighting to.
            lighting_level: Lighting level in Watts.
            replace: If True, replace existing lighting object in zone.

        Raises:
            ValueError: If zone not found in model.
            ValueError: If lighting already exists in zone and replace is False.

        Examples:
            >>> model.add_lighting("Zone1", 10)
        """
        if self.zone is None:
            raise ValueError("Zone not found in model.")
        if zone not in self.zone:
            raise ValueError(f"{zone} not found in model.")
        if self.lights is None:
            raise ValueError("Lights not found in model.")
        dict2 = copy.deepcopy(self.lights)

        if self.lights is not None:
            for k, v in dict2.items():
                if v.zone_or_zonelist_or_space_or_spacelist_name == zone:
                    if replace:
                        del self.lights[k]
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
            "constant_on",
            epm.ScheduleConstant(
                schedule_type_limits_name="on_off",
                hourly_value=1,
            ),
        )

        # Add lighting to epjs dictionary
        self.add(
            "lights",
            zone,
            epm.Lights(
                design_level_calculation_method=epm.DesignLevelCalculationMethod.lighting_level,
                fraction_radiant=0,
                fraction_replaceable=1,
                fraction_visible=1,
                lighting_level=lighting_level,
                return_air_fraction=0,
                schedule_name="constant_on",
                zone_or_zonelist_or_space_or_spacelist_name=zone,
            ),
        )

    def add_output(
        self,
        output_type: str,
        output_name: str,
        reporting_frequency: str = "Timestep",
    ):
        """Add an output variable or meter to the epjs dictionary.

        Args:
            output_type: Type of the output. "variable" or "meter".
            output_name: Name of the output variable or meter.
            reporting_frequency: Reporting frequency of the output variable or meter.

        Raises:
            ValueError: If output_type is not "variable" or "meter".

        Examples:
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
