import frads as fr

# Initialize an EnergyPlus model with an input of idf or epjs file
epmodel = fr.EnergyPlusModel("RefBldgMediumOfficeNew2004_southzone.idf")

# Add glazing systems (CFS) to EnergyPlus model
# 4 glazing systems for the 4 electrochromatic tinted states
gs_ec01 = fr.GlazingSystem()
gs_ec01.add_glazing_layer(
    "igsdb_product_7405.json"
)  # SageGlass SR2.0_7mm lami fully tinted 1%T
gs_ec01.add_glazing_layer("CLEAR_3.DAT")
gs_ec01.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
gs_ec01.name = "ec01"

gs_ec06 = fr.GlazingSystem()
gs_ec06.add_glazing_layer(
    "igsdb_product_7407.json"
)  # SageGlass® SR2.0_7mm lami int state 6%T
gs_ec06.add_glazing_layer("CLEAR_3.DAT")
gs_ec06.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
gs_ec06.name = "ec06"

gs_ec18 = fr.GlazingSystem()
gs_ec18.add_glazing_layer(
    "igsdb_product_7404.json"
)  # SageGlass® SR2.0_7mm lami int state 18%T
gs_ec18.add_glazing_layer("CLEAR_3.DAT")
gs_ec18.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
gs_ec18.name = "ec18"

gs_ec60 = fr.GlazingSystem()
gs_ec60.add_glazing_layer(
    "igsdb_product_7406.json"
)  # SageGlass® SR2.0_7mm lami full clear 60%T
gs_ec60.add_glazing_layer("CLEAR_3.DAT")
gs_ec60.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
gs_ec60.name = "ec60"

# Add glazing systems to EnergyPlus model
epmodel.add_glazing_system(gs_ec01)
epmodel.add_glazing_system(gs_ec06)
epmodel.add_glazing_system(gs_ec18)
epmodel.add_glazing_system(gs_ec60)

# Add lighting system to EnergyPlus model
epmodel.add_lighting("Perimeter_bot_ZN_1", replace=True)

# Build a Radiance model from the EnergyPlus model
radmodel = fr.epjson_to_rad(epmodel, epw="USA_CA_Oakland.Intl.AP.724930_TMY3.epw")

# Generate matrices for the Three-Phase method. 
# The matrices are used to calculate workplane illuminance 
# used in the controller function
rad_cfg = fr.WorkflowConfig.from_dict(radmodel["Perimeter_bot_ZN_1"])
rad_workflow = fr.ThreePhaseMethod(rad_cfg)
rad_workflow.generate_matrices()
tmx_dict = {
    "ec01": fr.load_matrix("Resources/ec01.xml"),
    "ec06": fr.load_matrix("Resources/ec06.xml"),
    "ec18": fr.load_matrix("Resources/ec18.xml"),
    "ec60": fr.load_matrix("Resources/ec60.xml"),
}

# Define controller function for
# facade shading state
# electric lighting power level
# Cooling setpoint temperature
def controller(state):
    if not epmodel.api.exchange.api_data_fully_ready(state):
        return
    # control facade shading state base on exterior solar irradiance
    # get exterior solar irradiance
    ext_irradiance = ep.get_variable_value(
        name="Surface Outside Face Incident Solar Radiation Rate per Area",
        key="Perimeter_bot_ZN_1_Wall_South_Window",
    )
    # facade shading state control algorithm
    if ext_irradiance <= 300:
        ec = "60"
    elif ext_irradiance <= 400 and ext_irradiance > 300:
        ec = "18"
    elif ext_irradiance <= 450 and ext_irradiance > 400:
        ec = "06"
    elif ext_irradiance > 450:
        ec = "01"
    shade = f"ec{ec}"
    # actuate facade shading state
    ep.actuate(
        component_type="Surface",
        name="Construction State",
        key="Perimeter_bot_ZN_1_Wall_South_Window",
        value=ep.construction_handles[shade],
    )

    # control cooling setpoint temperature based on time of day
    # pre-cooling
    # get current time
    datetime = ep.get_datetime()
    # control cooling setpoint temperature control algorithm
    if datetime.hour >= 16 and datetime.hour < 21:
        clg_setpoint = 25.56
    elif datetime.hour >= 12 and datetime.hour < 16:
        clg_setpoint = 21.67
    else:
        clg_setpoint = 24.44
    # actuate cooling setpoint temperature
    ep.actuate(
        component_type="Zone Temperature Control",
        name="Cooling Setpoint",
        key="PERIMETER_BOT_ZN_1",
        value=clg_setpoint,
    )

    # control electric lighting power based on occupancy and workplane illuminance
    # daylight dimming
    # get occupant count and direct and diffuse solar irradiance
    occupant_count = ep.get_variable_value(
        name="Zone People Occupant Count",
        key="PERIMETER_BOT_ZN_1"
    )
    direct_normal_irradiance = ep.get_variable_value(
        name="Site Direct Solar Radiation Rate per Area",
        key="Environment"
    )
    diffuse_horizontal_irradiance = ep.get_variable_value(
        name="Site Diffuse Solar Radiation Rate per Area",
        key="Environment"
    )
    # calculate average workplane illuminance
    avg_wpi = rad_workflow.calculate_sensor(
        "Perimeter_bot_ZN_1_Perimeter_bot_ZN_1_Floor",
        tmx_dict[shade],
        datetime,
        direct_normal_irradiance,
        diffuse_horizontal_irradiance,
    ).mean()
    # electric lighting power control algorithm
    if occupant_count > 0:
        lighting_power = (
            1 - min(avg_wpi / 500, 1)
        ) * 1200  # 1200W is the nominal lighting power density
    else:
        lighting_power = 0
    # actuate electric lighting power
    ep.actuate(
        component_type="Lights",
        name="Electricity Rate",
        key="Light_Perimeter_bot_ZN_1",
        value=lighting_power,
    )


# Add output
epmodel.add_output(
    output_type="variable",
    name="Surface Window Transmitted Solar Radiation Rate",
)
epmodel.add_output(
    output_type="variable", 
    name="Zone Lights Electricity Rate"
)
epmodel.add_output(
    output_type="variable", 
    name="Zone Thermostat Cooling Setpoint Temperature"
)
epmodel.add_output(
    output_type="variable",
    name="Zone People Occupant Count"
)
epmodel.add_output(
    output_type="variable",
    name="Surface Outside Face Incident Solar Radiation Rate per Area"
)

# ## Run Simulation
with fr.EnergyPlusSetup(
    epmodel, weather_file="USA_CA_Oakland.Intl.AP.724930_TMY3.epw"
) as ep:
    # request variables to be accessible during simulation
    ep.request_variable(
        name="Site Direct Solar Radiation Rate per Area",
        key="Environment"
    )
    ep.request_variable(
        name="Site Diffuse Solar Radiation Rate per Area",
        key="Environment"
    )
    ep.request_variable(
        name="Zone People Occupant Count",
        key="PERIMETER_BOT_ZN_1"
    )
    ep.request_variable(
        name="Surface Outside Face Incident Solar Radiation Rate per Area",
        key="Perimeter_bot_ZN_1_Wall_South_Window",
    )

    # set controller function to be called at the beginning of each system timestep
    ep.set_callback("callback_begin_system_timestep_before_predictor", controller)

    # run simulation
    ep.run()
