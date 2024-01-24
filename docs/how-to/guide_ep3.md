# How to set up a callback function in EnergyPlus?

This guide will show you how to use the callback function to modify the EnergyPlus model during the simulation. 

The demonstration will use the callback function to change the cooling setpoint temperature based on time of the day or occupancy count at the beginning of each time step during runtime.

The callback function is a Python function that can only takes in `state` as the argument. The callback function is where you define the control logic. 

Use `EnergyPlusModel.set_callback()` to set up a callback function. The function takes in the calling point and the callback function. The callback function is called at each time step at the calling point. See [Application Guide for EMS](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v22.1.0/EMSApplicationGuide.pdf) for details about the various calling points. 

## 0. Import required Python libraries

```python
import frads as fr
from pyenergyplus.dataset import ref_models, weather_files
```

## 1. Initialize an EnergyPlus model

You will need a working EnergyPlus model in idf or epjson format to initialize an EnergyPlus model. Or you can load an EnergyPlus reference model from `pyenergyplus.dataset`. See [How to run a simple EnergyPlus simulation?](guide_ep1.md) for more information on how to setup an EnergyPlus model.

```python
epmodel = fr.load_energyplus_model(ref_models["medium_office"]) # (1)
```

1.  EnergyPlus medium size office reference model from `pyenergyplus.dataset`.

## 2. Initialize EnergyPlus Simulation Setup

Initialize EnergyPlus simulation setup by calling `EnergyPlusSetup` and passing in an EnergyPlus model and an optional weather file.

```python
epsetup = fr.EnergyPlusSetup(
    epmodel, weather_files["usa_ca_san_francisco"], enable_radiance=True
) # (1)
```

1.  San Francisco, CA weather file from `pyenergyplus.dataset`.

## 3. Define the callback function

Before going into the control logic defined in the callback function, you need to first check if the api is ready at the beginning of each time step.

```python
def controller(state):
# check if the api is fully ready
    if not epsetup.api.exchange.api_data_fully_ready(state):
        return
```

### Update EnergyPlus model

Use `EnergyPlusSetup.actuate` to set or update the operating value of an actuator in the EnergyPlus model. `EnergyPlusSetup.actuate` takes in a component type, name, key, and value. The component type is the actuator category, e.g. "Weather Data". The name is the name of the actuator, e.g. "Outdoor Dew Point". The key is the instance of the variable to retrieve, e.g. "Environment". The value is the value to set the actuator to.

!!! tip
    Use `EnergyPlusSetup.actuators` to get a list of actuators in the EnergyPlus model. [component type, name, key]

There are also built-in actuator in frads that allows easier actuation of common actuators. See [Built-in Actuators](../ref/eplus.md/#frads.EnergyPlusSetup.actuate_cfs_state) for more details.

* `EnergyPlusSetup.actuate_cfs_state`
* `EnergyPlusSetup.actuate_heating_setpoint`
* `EnergyPlusSetup.actuate_cooling_setpoint`
* `EnergyPlusSetup.actuate_lighting_power`

First, get the current time from the EnergyPlus model by using `EnergyPlusSetup.get_datetime`. If the current time is between 9 am and 5 pm, set the cooling setpoint to 21 degree Celsius. Otherwise, set the cooling setpoint to 24 degree Celsius.

=== "EnergyPlusSetup.actuate"

    ```python 
    def controller(state):
        # check if the api is fully ready
        if not epsetup.api.exchange.api_data_fully_ready(state):
            return
        # get the current time
        datetime = epsetup.get_datetime()
        if datetime.hour > 9 and datetime.hour < 17:
            epsetup.actuate(
            component_type="Zone Temperature Control",
            name="Cooling Setpoint",
            key="Perimeter_bot_ZN_1",
            value=21,
        )
        else:
            epsetup.actuate(
            component_type="Zone Temperature Control",
            name="Cooling Setpoint",
            key="Perimeter_bot_ZN_1",
            value=24,
        )
    ```

=== "EnergyPlusSetup.actuate_cooling_setpoint"

    ```python
    def controller(state):
        # check if the api is fully ready
        if not epsetup.api.exchange.api_data_fully_ready(state):
            return
        # get the current time
        datetime = epsetup.get_datetime()
        if datetime.hour > 9 and datetime.hour < 17:
            epsetup.actuate_cooling_setpoint(zone="Perimeter_bot_ZN_1", value=21)
        else:
            epsetup.actuate_cooling_setpoint(zone="Perimeter_bot_ZN_1", value=24)
    ```

### Access EnergyPlus variable

Access EnergyPlus variable during simulation by using `EnergyPlusSetup.get_variable_value` and passing in a variable name and key.

!!! tip 
    Use `EnergyPlusSetup.get_variable_value` to access the EnergyPlus variable during the simulation and use the variable as a control input. 

Use `EnergyPlusSetup.get_variable_value` to get the current number of occupants in the zone. If the number of occupants is greater than 0, set the cooling setpoint to 21 degree Celsius. Otherwise, set the cooling setpoint to 24 degree Celsius.

```python
def controller(state):
    # check if the api is fully ready
    if not epsetup.api.exchange.api_data_fully_ready(state):
        return
    # get the current number of occupants in the zone
    num_occupants = epsetup.get_variable_value(
        variable_name="Zone People Occupant Count",
        key="Perimeter_bot_ZN_1",
    )
    if num_occupants > 0:
        epsetup.actuate_cooling_setpoint(zone="Perimeter_bot_ZN_1", value=21)
    else:
        epsetup.actuate_cooling_setpoint(zone="Perimeter_bot_ZN_1", value=24)
```

## 4. Set callback

Use `EnergyPlusModel.set_callback()` to set up a callback function. The example uses `callback_begin_system_timestep_before_predictor`.

!!! quote "BeginTimestepBeforePredictor"
    The calling point called “BeginTimestepBeforePredictor” occurs near the beginning of each timestep but before the predictor executes. “Predictor” refers to the step in EnergyPlus modeling when the zone loads are calculated. This calling point is useful for controlling components that affect the thermal loads the HVAC systems will then attempt to meet. Programs called from this point might actuate internal gains based on current weather or on the results from the previous timestep. Demand management routines might use this calling point to reduce lighting or process loads, change thermostat settings, etc.

```Python
epsetup.set_callback(
    "callback_begin_system_timestep_before_predictor",
    controller
)
```

## 5. Run the EnergyPlus simulation

```python
epsetup.run()
```