# How to model dynamic shading control and daylight dimming with EnergyPlus?

This guide will demonstrate how to use a controller function to control the shading state, cooling setpoint temperature, and electric lighting power intensity during simulation. 

The example is a medium office building with a four tinted states electrochromic glazing system. At the beginning of each timestep, EnergyPlus will call the controller function that operates the facade shading state based on exterior solar irradiance, cooling setpoint temperature based on time of day (pre-cooling), and electric lighting power intensity based on occupancy and workplane illuminance (daylight dimming). The workplane illuminance is calculated using the three-phase method in Radiance.

**Workflow**

1. [Setup an EnergyPlus Model](#1-setup-an-energyplus-model)

    1.1 [Initialize an EnergyPlus model](#11-initialize-an-energyplus-model)

    1.2 [Create glazing systems (Complex Fenestration States)](#12-create-glazing-systems-complex-fenestration-states)

    1.3 [Add glazing systems to EnergyPlus model](#13-add-glazing-systems-to-energyplus-model)

    1.4 [Add lighting systems to EnergyPlus model](#14-add-lighting-systems-to-energyplus-model)

2. [Setup EnergyPlus Simulation](#2-setup-energyplus-simulation)

    2.1 [Initialize EnergyPlus Simulation Setup](#21-initialize-energyplus-simulation-setup)

    2.2 [Define control algorithms using a controller function](#22-define-control-algorithms-using-a-controller-function)

    2.3 [Set callback](#23-set-callback)

    2.4 [Run simulation](#24-run-simulation)


``` mermaid
graph LR
    subgraph IGSDB
    A[Step 1.2 <br/> glazing/shading products];
    end

    subgraph frads

    C[Step 1.1 idf/epjs] --> |Initialize an EnergyPlus model| E;

    subgraph Step 2 EnergyPlus Simulation Setup
    subgraph Radiance
    R[Workplane Illuminance];
    end
    subgraph EnergyPlus
    E[EnergyPlusModel]<--> K[Step 2.2 & 2.3 <br/> controller function];
    E <--> R;
    K <--> R;
    end
    end

    subgraph  WincalcEngine
    A --> D[Step 1.3 <br/> create a glazing system <br/> per CFS state];
    D --> |Add glazing systems| E;
    end

    L[Step 1.4 lighting systems] --> |Add lighting| E;

    end
```

## 0. Import required Python libraries

```python
import frads as fr
from pyenergyplus.dataset import ref_models, weather_files
```

## 1. Setup an EnergyPlus Model
### 1.1 Initialize an EnergyPlus model

You will need a working EnergyPlus model in idf or epjson format to initialize an EnergyPlus model. Or you can load an EnergyPlus reference model from `pyenergyplus.dataset`. See [How to run a simple EnergyPlus simulation?](guide_ep1.md) for more information on how to setup an EnergyPlus model.

```python
epmodel = fr.load_energyplus_model(ref_models["medium_office"]) # (1)
```

1.  EnergyPlus medium size office reference model from `pyenergyplus.dataset`.

### 1.2 Create glazing systems (Complex Fenestration States)

!!! example "Create four glazing systems for the four electrochromic tinted states"
    Each glazing system consists of:

    * One layer of electrochromic glass
    * One gap (10% air and 90% argon) at 0.0127 m thickness
    * One layer of clear glass

Call `create_glazing_system` to create a glazing system. See [How to create a glazing system?](guide_ep2.md) for more details on how to create a glazing system.

```python 
gs_ec01 = fr.create_glazing_system(
    name="ec01",
    layers=[
        "igsdb_product_7405.json", # electrochromic glass Tvis: 0.01
        "igsdb_product_364.json", # clear glass
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
            "igsdb_product_7407.json", 
            "igsdb_product_364.json",
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
            "igsdb_product_7404.json",
            "igsdb_product_364.json",
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
            "igsdb_product_7406.json",
            "igsdb_product_364.json",
        ],
        gaps=[
            fr.Gap(
                [fr.Gas("air", 0.1), fr.Gas("argon", 0.9)], 0.0127
            )
        ],
    )
    ```

### 1.3 Add glazing systems to EnergyPlus model

Call `EnergyPlusModel.add_glazing_system()` to add glazing systems to the EnergyPlus model. 

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
Call `EnergyPlusModel.add_lighting` to add lighting systems to the EnergyPlus model. 

```python
epmodel.add_lighting(
    zone="Perimeter_bot_ZN_1",
    lighting_level=1200, # (1)
    replace=True
)
```

1. 1200W is the maximum lighting power density for the zone. This will be dimmed based on the daylight illuminance.

## 2. Setup EnergyPlus Simulation

### 2.1 Initialize EnergyPlus Simulation Setup

Initialize EnergyPlus simulation setup by calling `EnergyPlusSetup` and passing in an EnergyPlus model and an optional weather file. Enable Radiance for daylighting simulation by setting `enable_radiance` to `True`. See [How to enable Radiance in EnergyPlus simulation?](guide_radep1.md) for more information.

```python
epsetup = fr.EnergyPlusSetup(
    epmodel, weather_files["usa_ca_san_francisco"], enable_radiance=True
) # (1)
```

1.  San Francisco, CA weather file from `pyenergyplus.dataset`.

### 2.2 Define control algorithms using a controller function

The controller function defines the control algorithm and control the facade shading state, cooling setpoint temperature, and electric lighting power intensity in the EnergyPlus model during simulation.

!!! example "Controller function"
    The example shows how to implement control algorithms for zone "Perimeter_bot_ZN_1", which has a window named "Perimeter_bot_ZN_1_Wall_South_Window" and lighting named "Perimeter_bot_ZN_1".

    * **Facade CFS state** based on exterior solar irradiance
    * **Cooling setpoint temperature** based on time of day (pre-cooling)
    * **Electric lighting power intensity** based on occupancy and workplane illuminance (daylight dimming)


The controller function takes in a `state` argument. See [How to set up a callback function in EnergyPlus?](guide_ep3.md) for more details on how to define a controller function.

```py linenums="1" hl_lines="2 6 28 44"

def controller(state):
    # check if the api is fully ready
    if not epsetup.api.exchange.api_data_fully_ready(state):
        return

    # control facade shading state based on exterior solar irradiance
    # get exterior solar irradiance
    ext_irradiance = epsetup.get_variable_value(
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
    epsetup.actuate_cfs_state(
        window="Perimeter_bot_ZN_1_Wall_South_Window",
        cfs_state=cfs_state,
    )

    # control cooling setpoint temperature based on the time of day
    # pre-cooling
    # get the current time
    datetime = epsetup.get_datetime()
    # cooling setpoint temperature control algorithm
    if datetime.hour >= 16 and datetime.hour < 21:
        clg_setpoint = 25.56
    elif datetime.hour >= 12 and datetime.hour < 16:
        clg_setpoint = 21.67
    else:
        clg_setpoint = 24.44
    # actuate cooling setpoint temperature
    epsetup.actuate_cooling_setpoint(
        zone="Perimeter_bot_ZN_1", value=clg_setpoint
    )

    # control lighting power based on occupancy and workplane illuminance
    # daylight dimming
    # get occupant count and direct and diffuse solar irradiance
    occupant_count = epsetup.get_variable_value(
        name="Zone People Occupant Count", key="PERIMETER_BOT_ZN_1"
    )
    # calculate average workplane illuminance using Radiance
    avg_wpi = epsetup.calculate_wpi(
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
    epsetup.actuate_lighting_power(
        light="Perimeter_bot_ZN_1",
        value=lighting_power,
    )
```

### 2.3 Set callback

Register the controller functions to be called back by EnergyPlus during runtime by calling `set_callback`and passing in a callback point and function. See [How to set up a callback function in EnergyPlus?](guide_ep3.md) for more details.

```Python
epsetup.set_callback(
    "callback_begin_system_timestep_before_predictor",
    controller
)
```

### 2.4 Run simulation

```python
epsetup.run()
```
