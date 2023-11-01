# How to model dynamic shading control and daylight dimming with EnergyPlus?


The example demonstrates how to use a controller function to control the shading state, cooling setpoint temperature, and electric lighting power intensity during simulation. At the beginning of each timestep, EnergyPlus will call the controller function that operates the facade shading state based on exterior solar irradiance, cooling setpoint temperature based on time of day (pre-cooling), and electric lighting power intensity based on occupancy and workplane illuminance (daylight dimming). The workplane illuminance is calculated using the three-phase method through Radiance.

**Workflow**

1. [Setup an EnergyPlus Model](#1-setup-an-energyplus-model)

2. [Setup EnergyPlus Simulation](#2-setup-energyplus-simulation)


```mermaid
graph LR

    subgraph <b>IGSDB</b>
    A[Step 1.2 glazing products]
    B[Step 1.2 shading products]
    end

    subgraph <b>frads</b>

    C[Step 1.1 idf/epjs] --> |Initialize an EnergyPlus model| E;

    subgraph Step 2 EnergyPlus Simulation Setup
    subgraph <b>Radiance</b>
    R[Workplane Illuminance]
    end
    subgraph <b>EnergyPlus</b>
    E[EnergyPlusModel]<--> K[Step 2.2 & 2.3 controller function<br/> <br/> * switch shading state <br/> * daylight dimming <br/>* pre-cooling <br/>]
    E <--> R
    K <--> R;
    end
    end

    subgraph  <b>WincalcEngine</b>
    A --> D[Step 1.3 glazing/shading system<br/>for each CFS state];
    B --> D;
    D --> |Add glazing systems| E;
    end

    L[Step 1.4 lighting systems] --> |Add lighting| E;

    end
```

## 0. Import required Python libraries

```python
from pathlib import Path
import frads as fr

```

!!! tip "Tips: Reference EnergyPlus models and weather files"
    The `pyenergyplus.dataset` module contains a dictionary of EnergyPlus models and weather files. The keys are the names of the models and weather files. The values are the file paths to the models and weather files.

    ```
    from pyenergyplus.dataset import ref_models, weather_files
    ```


## 1. Setup an EnergyPlus Model
### 1.1 Initialize an EnergyPlus model

Initialize an EnergyPlus model by calling `load_energyplus_model` and passing in an EnergyPlus model in an idf or epjson file format.

```python
epmodel = fr.load_energyplus_model(ref_models["medium_office"])
```

or

```python
epmodel = fr.load_energyplus_model("medium_office.idf")
```

### 1.2 Create glazing systems (Complex Fenestration States)

!!! example "Create four glazing systems for the four electrochromic tinted states"
    Each glazing system consists of:

    * One layer of electrochromic glass
    * One gap (10% air and 90% argon) at 0.0127 m thickness
    * One layer of clear glass

Create a glazing system by calling `create_glazing_system`, which returns a `GlazingSystem` object. `create_glazing_system` takes in the following arguments:

* `name`: the name of the glazing system.
* `layers`: a list of file paths to the glazing or shading layers in the glazing system, in order from exterior to interior. Visit the [IGSDB](https://igsdb.lbl.gov/) website to download  `.json` files for glazing products and `.xml` files for shading products.
* `gaps`: a list of `Gap` objects. Each `Gap` object consists of a list of `Gas` objects and a float defining the gap thickness. The `Gas` object consists of the gas type and the gas fraction. The gas fraction is a float between 0 and 1. The default gap is air at 0.0127 m thickness.

```python
gs_ec01 = fr.create_glazing_system(
    name="ec01",
    layers=[
        Path("products/igsdb_product_7405.json"),
        Path("products/CLEAR_3.DAT"),
    ],
    gaps=[
        fr.Gap(
            [fr.Gas("air", 0.1), fr.Gas("argon", 0.9)], 0.0127
        )
    ],
)
```

??? info "Create glazing systems for the other tinted electrochromic states"
    ```python
    gs_ec06 = fr.create_glazing_system(
        name="ec06",
        layers=[
            Path("products/igsdb_product_7407.json"),
            Path("products/CLEAR_3.DAT"),
        ],
        gaps=[
            fr.Gap(
                [fr.Gas("air", 0.1), fr.Gas("argon", 0.9)], 0.0127
            )
        ],
    )

    gs_ec18 = fr.create_glazing_system(
        name="ec18",
        layers=[
            Path("products/igsdb_product_7404.json"),
            Path("products/CLEAR_3.DAT"),
        ],
        gaps=[
            fr.Gap(
                [fr.Gas("air", 0.1), fr.Gas("argon", 0.9)], 0.0127
            )
        ],
    )

    gs_ec60 = fr.create_glazing_system(
        name="ec60",
        layers=[
            Path("products/igsdb_product_7406.json"),
            Path("products/CLEAR_3.DAT"),
        ],
        gaps=[
            fr.Gap(
                [fr.Gas("air", 0.1), fr.Gas("argon", 0.9)], 0.0127
            )
        ],
    )
    ```

### 1.3 Add glazing systems to EnergyPlus model

Call `add_glazing_system` from the `EnergyPlusModel` class to add glazing systems to the EnergyPlus model. `add_glazing_system` takes in a `GlazingSystem` object.


```python
epmodel.add_glazing_system(gs_ec01)
```
??? info "Add other glazing systems to the EnergyPlus model"
    ```python
    epmodel.add_glazing_system(gs_ec06)
    epmodel.add_glazing_system(gs_ec18)
    epmodel.add_glazing_system(gs_ec60)
    ```

### 1.4 Add lighting systems to EnergyPlus model
Call `add_lighting` from the `EnergyPlusModel` class to add lighting systems to the EnergyPlus model. `add_lighting` takes in the name of the zone to add lighting to and an optional `replace` argument. If `replace` is `True`, the zone's existing lighting system will be replaced by the new lighting system. If `replace` is `False` and the zone already has a lighting system, an error will be raised. The default value of `replace` is `False`.

```python
epmodel.add_lighting(
    zone="Perimeter_bot_ZN_1",
    replace=True
)
```

## 2. Setup EnergyPlus Simulation

### 2.1 Initialize EnergyPlus Simulation Setup

Initialize EnergyPlus simulation setup by calling `EnergyPlusSetup` and passing in an EnergyPlus model and an optional weather file.

To enable Radiance for daylighting simulation, set `enable_radiance` to `True`. The default value of `enable_radiance` is `False`.

```python
eps = fr.EnergyPlusSetup(
    epmodel, weather_files["usa_ca_san_francisco"], enable_radiance=True
)
```

### 2.2 Define control algorithms using a controller function

The controller function will control the facade shading state, cooling setpoint temperature, and electric lighting power intensity in the EnergyPlus model during simulation.

!!! example "Controller function"
    The example shows how to implement control algorithms for zone "Perimeter_bot_ZN_1", which has window "Perimeter_bot_ZN_1_Wall_South_Window" and lighting "Perimeter_bot_ZN_1_Lights".

    * **Facade CFS state** based on exterior solar irradiance
    * **Cooling setpoint temperature** based on time of day (pre-cooling)
    * **Electric lighting power intensity** based on occupancy and workplane illuminance (daylight dimming)

!!! notes "Actuate"
    * **Generic actuator**

    Use `EnergyPlusSetup.actuate` to set or update the operating value of an actuator in the EnergyPlus model. `EnergyPlusSetup.actuate` takes in a component type, name, key, and value. The component type is the actuator category, e.g. "Weather Data". The name is the name of the actuator, e.g. "Outdoor Dew Point". The key is the instance of the variable to retrieve, e.g. "Environment". The value is the value to set the actuator to.

    * **Special actuator**

        * **Facade CFS state**

        `EnergyPlusSetup.actuate_cfs_state` takes in a window name and a CFS state (the name of the glazing system).

        * **Heating/Cooling setpoint temperature**

        `EnergyPlusSetup.actuate_heating_setpoint` takes in a zone name and a heating setpoint temperature.
        `EnergyPlusSetup.actuate_cooling_setpoint` takes in a zone name and a cooling setpoint temperature.

        * **Electric lighting power intensity**

        `EnergyPlusSetup.actuate_lighting_power` takes in a lighting name and a lighting power intensity.

!!! notes "Get variable value"

    Access EnergyPlus variable during simulation by using `EnergyPlusSetup.get_variable_value` and passing in a variable name and key

    !!! tip "Tips"
        Use `EnergyPlusSetup.get_variable_value` to access the EnergyPlus variable during the simulation and use the variable as a control input. For example, use the exterior solar irradiance to control the facade CFS state.

The controller function takes in a `state` argument.

```py linenums="1" hl_lines="2 6 28 44"

def controller(state):
    # check if the api is fully ready
    if not eps.api.exchange.api_data_fully_ready(state):
        return

    # control facade shading state based on exterior solar irradiance
    # get exterior solar irradiance
    ext_irradiance = eps.get_variable_value(
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
    cfs_state = f"ec{ec}"
    # actuate facade shading state
    eps.actuate_cfs_state(
        window="Perimeter_bot_ZN_1_Wall_South_Window",
        cfs_state=cfs_state,
    )

    # control cooling setpoint temperature based on the time of day
    # pre-cooling
    # get the current time
    datetime = ep.get_datetime()
    # cooling setpoint temperature control algorithm
    if datetime.hour >= 16 and datetime.hour < 21:
        clg_setpoint = 25.56
    elif datetime.hour >= 12 and datetime.hour < 16:
        clg_setpoint = 21.67
    else:
        clg_setpoint = 24.44
    # actuate cooling setpoint temperature
    eps.actuate_cooling_setpoint(
        zone="Perimeter_bot_ZN_1", value=clg_setpoint
    )

    # control lighting power based on occupancy and workplane illuminance
    # daylight dimming
    # get occupant count and direct and diffuse solar irradiance
    occupant_count = eps.get_variable_value(
        name="Zone People Occupant Count", key="PERIMETER_BOT_ZN_1"
    )
    # calculate average workplane illuminance using Radiance
    avg_wpi = eps.calculate_wpi(
        zone="Perimeter_bot_ZN_1",
        cfs_name={
            "Perimeter_bot_ZN_1_Wall_South_Window": cfs_state
        },
    ).mean()
    # electric lighting power control algorithm
    if occupant_count > 0:
        lighting_power = (
            1 - min(avg_wpi / 500, 1)
        ) * 1200  # 1200W is the nominal lighting power density
    else:
        lighting_power = 0
    # actuate electric lighting power
    eps.actuate_lighting_power(
        light="Perimeter_bot_ZN_1_Lights",
        value=lighting_power,
    )
```

### 2.3 Set callback

Register the controller functions to be called back by EnergyPlus during runtime by calling `set_callback`and passing in a callback point and function. Refer to [Application Guide for EMS](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v22.1.0/EMSApplicationGuide.pdf) for descriptions of the calling points.

This example uses `callback_begin_system_timestep_before_predictor`.

!!! quote "BeginTimestepBeforePredictor"
    The calling point called “BeginTimestepBeforePredictor” occurs near the beginning of each timestep but before the predictor executes. “Predictor” refers to the step in EnergyPlus modeling when the zone loads are calculated. This calling point is useful for controlling components that affect the thermal loads the HVAC systems will then attempt to meet. Programs called from this point might actuate internal gains based on current weather or on the results from the previous timestep. Demand management routines might use this calling point to reduce lighting or process loads, change thermostat settings, etc.

```Python
eps.set_callback("callback_begin_system_timestep_before_predictor", controller)

```

### 2.4 Run simulation

To simulate, use `run` with optional parameters:

* output_directory: Output directory path. (default: current directory)
* output_prefix: Prefix for output files. (default: eplus)
* output_suffix: Suffix style for output files. (default: L)
    * L: Legacy (e.g., eplustbl.csv)
    * C: Capital (e.g., eplusTable.csv)
    * D: Dash (e.g., eplus-table.csv)
* silent: If True, do not print EnergyPlus output to console. (default: False)
* annual: If True, force run annual simulation. (default: False)
* design_day: If True, force run design-day-only simulation. (default: False)


```python
eps.run()
```
