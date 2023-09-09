# How to model dynamic shading control and daylight dimming with EnergyPlus?

This notebook shows how to use EnergyPlusAPI to 



You will also need a working EnergyPlus model in an idf or epjson file format. The model should have at least one exterior window.


## Workflow
1. [Initialize an EnergyPlus model with an input of idf or epjs file.](#initialize-an-energyplus-model)
2. [Create Complex Fenestration System (CFS) glazing system for each fenestration construction state and add to the EnergyPlus model.](#add-cfs-objects-to-the-energyplus-model)
3. [If implementing daylight dimming, create lighting objects and add to the EnergyPlus model.](#add-lighting-objects-to-the-energyplus-model)
4. [Define controller function for CFS construction states and daylight dimming](#define-controller-function-for-cfs-construction-states-and-daylight-dimming)
5. [Initialize pyenergyplus.api to simulate](#initialize-pyenergyplusapi-to-simulate)



```mermaid
graph LR

    subgraph <b>IGSDB</b>
    A[Step 2: glazing products]
    B[Step 2: shading products]
    end

    subgraph <b>frads</b>

    C[Step 1: idf/epjs] --> |Initialize an EnergyPlus model| E;

    subgraph Step 5: simulation
    subgraph <b>Radiance</b>
    R[Step 4: Workplane Illuminance]
    end
    subgraph <b>EnergyPlus</b>
    E[EPmodel]<--> K[Step 4: controller function:<br/> * switch CFS state<br/> * daylight dimming]
    E <--> R
    K <--> R;
    end
    end

    subgraph  <b>WincalcEngine</b>
    A --> D[Step 2: glazing/shading system<br/>for each CFS state];
    B --> D;
    D --> |Add CFS| E;
    end

    L[Step3: lighting system] --> |Add lighting| E;

    end
```

## Import required Python libraries

```python
import os
from pathlib import Path
import sys 

import frads as fr
import pandas as pd
```

# EnergyPlus Model
##  Initialize an EnergyPlus model
The [example idf](https://github.com/LBNL-ETA/frads/blob/main/test/Resources/RefBldgMediumOfficeNew2004_southzone.idf) based on the DOE commercial reference  medium office. This file has two zones: one perimeter zone with one south-facing window and one plenum zone.


Initialize an EnergyPlus model by calling `EnergyPlusModel` class with an input of idf or epjs file. 


```python
epmodel = fr.EnergyPlusModel(Path("RefBldgMediumOfficeNew2004_southzone.idf"))
```

## Create glazing systems (Complex Fenestration States)

!!! info "Create four glazing systems for the four electrochromatic tinted states."
    Each glazing system consists of one layer of electrochromic glass and one layer of clear glass. The gap between the glasses is 10% air and 90% argon.

Initialize a glazing system by calling `GlazingSystem()`.

Then, use `add_glazing_layer` and `add_shading_layer` to add glazing and shading layer to the glazing system. The layers should added from the outside to the inside.  `add_glazing_layer ` takes in a `.dat` or `.json` file. `add_shading_layer` takes in a `.xml` file. Visit the [IGSDB](https://igsdb.lbl.gov/) website to download  `.json` files for glazing products and `.xml` files for shading products. 

If not specified, the default gap between the layers is air at 0.0127 m thickness. To customize the gap, use the `gaps`attribute of the `GlazingSystem`class. The `gaps`attribute is a list of tuples. Each tuple consists of tuples, where the first item is the gas type and the second item is the gas ratio, and a float for the gap thickness. The default thickness is 0.0127 m. 

```python
gs_ec01 = fr.GlazingSystem()
gs_ec01.add_glazing_layer(
    "igsdb_product_7405.json"
)  # SageGlass SR2.0_7mm lami fully tinted 1%T
gs_ec01.add_glazing_layer("CLEAR_3.DAT") # 3mm clear glass
gs_ec01.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
```

!!! tip "Glazing system name"

    ```
    >>> gs_ec01_d.name # get glazing system name
     'igsdb_product_7405_Generic Clear Glass'

    >>> gs_ec01.name = "ec01" # customize glazing system name
    ```

??? example "Create glazing systems for the other tinted electrochromatic states"
    ```python
    gs_ec06 = fr.GlazingSystem()
    gs_ec06.add_glazing_layer(
        "products/igsdb_product_7407.json"
    )  # SageGlass® SR2.0_7mm lami int state 6%T
    gs_ec06.add_glazing_layer("products/CLEAR_3.DAT")
    gs_ec06.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
    gs_ec06.name = "ec06"

    gs_ec18 = fr.GlazingSystem()
    gs_ec18.add_glazing_layer(
        "products/igsdb_product_7404.json"
    )  # SageGlass® SR2.0_7mm lami int state 18%T
    gs_ec18.add_glazing_layer("products/CLEAR_3.DAT")
    gs_ec18.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
    gs_ec18.name = "ec18"

    gs_ec60 = fr.GlazingSystem()
    gs_ec60.add_glazing_layer(
        "products/igsdb_product_7406.json"
    )  # SageGlass® SR2.0_7mm lami full clear 60%T
    gs_ec60.add_glazing_layer("products/CLEAR_3.DAT")
    gs_ec60.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
    gs_ec60.name = "ec60"
    ```

## Add glazing systems to EnergyPlus model

Call `add_glazing_system` from the `EnergyPlusModel` class to add glazing systems to the EnergyPlus model.


```python
epmodel.add_glazing_system(gs_ec01)
```
??? example "Add other glazing systems to the EnergyPlus model"
    ```python
    epmodel.add_glazing_system(gs_ec06)
    epmodel.add_glazing_system(gs_ec18)
    epmodel.add_glazing_system(gs_ec60)
    ```

## Add lighting systems to EnergyPlus model
Call `add_lighting` from the `EnergyPlusModel` class to add lighting systems to the EnergyPlus model. `add_lighting` takes in the name of the zone the lighting system is in and an optional `replace` argument. If `replace` is `True`, the existing lighting system in the zone will be replaced by the new lighting system. If `replace` is `False` and the zone already has a lighting system, an error will be raised. The default value of `replace` is `False`.

```python
epmodel.add_lighting("Perimeter_bot_ZN_1", replace=True)
```

# Radiance Model

## Create a Radiance model from an EnergyPlus model

Create a Radiance model by calling `epjson_to_rad`and passing in an EnergyPlus model and an optional weather file.

```python
radmodel = fr.epjson_to_rad(epmodel, epw="USA_CA_Oakland.Intl.AP.724930_TMY3.epw")
```

## Use the Radiance model to perform the three-phase method for daylight simulation
Use `WorkflowConfig` to generate a workflow configuration for each zone that the three-phase method is computed in. The `WorkflowConfig.from_dict` method takes in a dictionary representing a zone. The dictionary can be accessed by calling `radmodel["ZoneName"]`.

```python
rad_cfg = fr.WorkflowConfig.from_dict(radmodel["Perimeter_bot_ZN_1"])
```


Use `ThreePhaseMethod`to perform the three-phase method. The `ThreePhaseMethod` class takes in a `WorkflowConfig` object.

```python
rad_workflow = fr.ThreePhaseMethod(rad_cfg)
```

Use `generate_matrices` to generate the view, daylight, and transmission matrices.

```python
rad_workflow.generate_matrices()
```

Use `load_matrices` to load transmission matrices. Then, create a dictionary of transmission matrices for each CFS state to be used in the controller function.

```python
tmx_dict = {
    "ec01": fr.load_matrix("ec01.xml"),
    "ec06": fr.load_matrix("ec06.xml"),
    "ec18": fr.load_matrix("ec18.xml"),
    "ec60": fr.load_matrix("ec60.xml"),
}
```

??? tip "list of attributes of EnergyPlusModel."

    ```
    >>> epmodel.zones
    ['FirstFloor_Plenum', 'Perimeter_bot_ZN_1']
    ```

    ```
    >>> epmodel.walls_window # wall with window
    ['Perimeter_bot_ZN_1_Wall_South']
    ```

    ```
    >>> epmodel.windows
    ['Perimeter_bot_ZN_1_Wall_South_Window']
    ```

    ```
    >>> epmodel.complex_fenestration_states
    ['ec01', 'ec06', 'ec18', 'ec60']
    ```

    ```
    >>> epmodel.lighting_zone
    ['Light_Perimeter_bot_ZN_1']
    ```

    ```
    >>> epmodel.floors
    ['FirstFloor_Plenum_Floor_1', 'Perimeter_bot_ZN_1_Floor']
    ```

# Define controller function 

!!! info "Control algorithms"
    * control facade shading state base on exterior solar irradiance
    * control cooling setpoint temperature based on time of day (pre-cooling)
    * control electric lighting power based on occupancy and workplane illuminance (daylight dimming)


**actuate**

Use `actuate` to set or update the operating value of an actuator in the EnergyPlus model. `actuate` takes in the component type, name, key, and value. The component type is the actuator category, e.g. "Weather Data". The name is the name of the actuator, e.g. "Outdoor Dew Point". The key is the instance of the variable to retrieve, e.g. "Environment". The value is the value to set the actuator to.

**variable**

You can access EnergyPlus variable during simulation using `get_variable_value` and passing in the same variable name and key. To access an Energyplus variable during simulation, you need to first request the variable before running the simulation by calling `request_variable` (more explanation below).

!!! tip 
    You can use `get_variable_value` to access the EnergyPlus variable during simulation and use the variable to control the actuator.

    For example, you can use `get_variable_value` to access the exterior solar irradiance and use the irradiance value to control the facade shading state.


```python
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
        name="Site Direct Solar Radiation Rate per Area", key="Environment"
    )
    diffuse_horizontal_irradiance = ep.get_variable_value(
        name="Site Diffuse Solar Radiation Rate per Area", key="Environment"
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
```

!!! note "Workplane illuminance calculation"
    rad_workflow.calculate_sensor() takes in the following arguments:
    * `sensor_name`: the name of the sensor
    * `tmx`: the transmission matrix of the CFS state
    * `datetime`: the datetime of the simulation
    * `direct_normal_irradiance`: the direct normal irradiance
    * `diffuse_horizontal_irradiance`: the diffuse horizontal irradiance

# Run simulation

## Add output variable

Use `add_output`to request output variables and meters that are not in the input idf file before the simulation run. `add_output` takes the name of the output variable or meter, the output type ("variable" or "meter"). It also takes in an optional argument `reporting_frequency` to specify the reporting frequency of the output variable or meter. The default value is "Hourly".

```python
epmodel.add_output(
    output_name="Zone Lights Electricity Rate",
    output_type="variable",
    reporting_frequency="timestep",
)
epmodel.add_output(
    output_name="Electricity:Facility", 
    output_type="meter"
)
```

## Initialize EnergyPlusSetup and run simulation

**Request variable**
To access a variable during the simulation, you need to first request the variable before running the simulation by calling `request_variable`and passing in a variable name and key. 

**Set callback**
Register the controller functions to be called back by EnergyPlus during runtime by calling `set_callback`and passing in a callback point and function. 

Refer to [Application Guide for EMS](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v22.1.0/EMSApplicationGuide.pdf) for descriptions of the calling points. This example uses `callback_begin_system_timestep_before_predictor`calling point to control the shading and lighting.

!!! quote "BeginTimestepBeforePredictor"
    The calling point called “BeginTimestepBeforePredictor” occurs near the beginning of each timestep but before the predictor executes. “Predictor” refers to the step in EnergyPlus modeling when the zone loads are calculated. This calling point is useful for controlling components that affect the thermal loads the HVAC systems will then attempt to meet. Programs called from this point might actuate internal gains based on current weather or on the results from the previous timestep. Demand management routines might use this calling point to reduce lighting or process loads, change thermostat settings, etc.

**Run simulation**
To simulate, use `run`with optional parameters:
    * output_directory: Output directory path. (default: None) If None, use current directory.
    * output_prefix: Prefix for output files. (default: eplus)
    * output_suffix: Suffix style for output files. (default: L)
        L: Legacy (e.g., eplustbl.csv)
        C: Capital (e.g., eplusTable.csv)
        D: Dash (e.g., eplus-table.csv)
    * silent: If True, do not print EnergyPlus output to console. (default: False)
    * annual: If True, force run annual simulation. (default: False)
    design_day: If True, force run design-day-only simulation. (default: False)


```python
with fr.EnergyPlusSetup(
    epmodel, weather_file="USA_CA_Oakland.Intl.AP.724930_TMY3.epw"
) as ep:
    # request variables to be accessible during simulation
    ep.request_variable(
        name="Site Direct Solar Radiation Rate per Area", key="Environment"
    )
    ep.request_variable(
        name="Site Diffuse Solar Radiation Rate per Area", key="Environment"
    )
    ep.request_variable(name="Zone People Occupant Count", key="PERIMETER_BOT_ZN_1")
    ep.request_variable(
        name="Surface Outside Face Incident Solar Radiation Rate per Area",
        key="Perimeter_bot_ZN_1_Wall_South_Window",
    )

    # set controller function to be called at the beginning of each system timestep
    ep.set_callback("callback_begin_system_timestep_before_predictor", controller)

    # run simulation
    ep.run()
```


### Load and visualize results
