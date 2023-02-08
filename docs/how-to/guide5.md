# How to model dynamic shading control and daylight dimming with EnergyPlus?

This notebook shows how to use EnergyPlusAPI to 

* switch the complex fenestration system (CFS) construction state based on time ([Example 1](#example-1-shading-control-and-daylight-dimming)) or direct normal irradiance ([Example 2](#example-2-electrochromic-glass-with-4-tinted-states))
* implement daylight dimming based on the workplane illuminance ([Example 1](#example-1-shading-control-and-daylight-dimming))

## Prerequisites

To run this notebook, you will need to install the following:

* [EnergyPlus](https://energyplus.net/) to simulate building energy. Version 9.2.0 or later is required.

Optional:

* [pandas](https://pandas.pydata.org/docs/getting_started/install.html)


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

!!! note 
    You'll need to install [EnergyPlus](https://www.energyplus.net) before proceeding.


Importing EnergyPlus into Python is not so straightforward. You can run the following code
to import EnergyPlusAPI if you had installed EnergyPlus in the default location for your
operating system. If you had not installed EnergyPlus in one of these locations you'll need to 
`append` the path to your `sys.path` before you can import EnergyPlusAPI. 

```python
srcloc = {'win32': 'C:\\', 'darwin': '/Applications', 'linux': '/usr/local'}
dname  = [os.path.join(srcloc[sys.platform], d) for d in os.listdir(srcloc[sys.platform]) if d.startswith('EnergyPlus')]
ephome = dname.pop()
if ephome not in sys.path:
    sys.path.append(ephome)

from pyenergyplus.api import EnergyPlusAPI
```

##  Initialize an EnergyPlus model
The [example idf](https://github.com/NREL/EnergyPlus/blob/develop/testfiles/1ZoneUncontrolled_win_1.idf) file is from the EnergyPlus ExampleFiles directory. The building is 15.24m X 15.24m, single zone with one south-facing window.


Initialize an EnergyPlus model by calling `load_epmodel`with an input of idf or epjs file. 


```python
idf_path = Path("1ZoneUncontrolled_win_1.idf")
api = EnergyPlusAPI()
epmodel = fr.load_epmodel(idf_path, api)
```
## Example 1 - shading control and daylight dimming

### Add CFS objects to the EnergyPlus model
Initialize a glazing system by calling `GlazingSystem()`.

Then, use `add_glazing_layer` and `add_shading_layer`respectively, to add glazing and shading layer to the glazing system. The layers should added from the outside to the inside.  `add_glazing_layer ` takes in a `.dat` or `.json` file. `add_shading_layer` takes in a `.xml` file. Visit the [IGSDB](https://igsdb.lbl.gov/) website to download  `.json` files for glazing products and `.xml` files for shading products. 

If not specified, the default gap between the layers is air at 0.0127 m thickness. See [Example 2](#example-2-electrochromic-glass-with-4-tinted-states) for how to customize a gap. 


Create an unshaded glazing system, consisting of one layer of 6 mm clear glass with the default gap. 


```python
gs_unshaded = fr.GlazingSystem()
gs_unshaded.add_glazing_layer("products/CLEAR_6.DAT")
```

!!! note annotate "This is how you get the name of the glazing system."

    ```
    >>> gs_unshaded.name
     'Generic Clear Glass'
    ```

Create a shaded glazing system, consisting of one layer of 6 mm clear glass and one layer of shading: 2011-SA1.


```python
gs_shaded = fr.GlazingSystem()
gs_shaded.add_glazing_layer("products/CLEAR_6.DAT")
gs_shaded.add_shading_layer("products/2011-SA1.XML")
```

After adding glazing and shading layers to the glazing system, compute solar and the solar and photopic results using `compute_solar_photopic_results`. Need to re-compute each time when the glazing system layering composition changes.


```python
gs_unshaded.compute_solar_photopic_results()
```


```python
gs_shaded.compute_solar_photopic_results()
```

Call `add_cfs` to add the unshaded and shaded glazing systems to the EnergyPlus model.


```python
epmodel.add_cfs(gs_unshaded)
epmodel.add_cfs(gs_shaded)
```

### Add lighting objects to the EnergyPlus model
Call `add_lighting`to add a lighting object for each of the zones in the building.

```python
epmodel.add_lighting()
```
!!! note annotate "This is a list of attributes of the Lighting class."

    ```
    >>> epmodel.windows
    ['Zn001:Wall001:Win001']
    ```

    ```
    >>> epmodel.walls_window
    ['Zn001:Wall001']
    ```

    ```
    >>> epmodel.floors
    ['Zn001:Flr001']
    ```

    ```
    >>> epmodel.cfs
    ['Generic Clear Glass', 'Generic Clear Glass_Satine 5500 5%, White Pearl']
    ```

    ```
    >>> epmodel.lighting_zone
    ['Light_ZONE ONE']
    ```
    ```
    >>> epmodel.zones
    ['ZONE ONE']
    ```

### Initialize a Radiance model
Create a Radiance model by calling `epjson2rad`and passing in an epjs and epw file. The epjs file can be accessed by calling `epmodel.epjs`.`epjson2rad` creates an `Objects`directory for material and geometry primitives and a `Resources`directory for transmission matrices (xml files). The `epjson2rad`function also generates a `config`file, which contains information about simulation controls setting, site, model, and raysender. 

Use `three_phase`to perform the three-phase method and generate the view and daylight matrices under the `Matrices`directory. 

Finally, use `load_matrix` to load the view, daylight, and transmission matrices.


```python
zone = epmodel.zones[0]
floor = epmodel.floors[0]
wall_wndo = epmodel.walls_window[0]

# generate Radiance model
# generate view, daylight, and transmission matrices
fr.epjson2rad(epmodel.epjs, epw="USA_CA_Oakland.Intl.AP.724930_TMY3.epw")
cfg_file = Path(f"{zone}.cfg")
config = fr.parse_mrad_config(cfg_file)
config["SimControl"]["no_multiply"] = "true"
with fr.assemble_model(config) as model:
    mpath = fr.three_phase(model, config)

# load matrices
vmx_window1 = fr.load_matrix(mpath.pvmx[f"{floor}{wall_wndo}_window"])
dmx_window1 = fr.load_matrix(mpath.dmx[f"{wall_wndo}_window"])
tmx_unshaded = fr.load_matrix(f"Resources/Generic Clear Glass.xml")
tmx_shaded = fr.load_matrix(
    f"Resources/Generic Clear Glass_Satine 5500 5%, White Pearl.xml"
)
```

### Define controller function for shading control and daylight dimming
Control fenestration construction based on time. The window is shaded from 11:00 to 15:00; otherwise, unshaded. The nominal lighting power of the light is 30W, controlled with linear daylight dimming based on the workplane illuminance. 

The workplane illuminance at each timestep by calling `multiply_rgb`and passing in the view, transmission, daylight, and sky matrices. Sky matrix is computed at each timestep by calling `genskymtx`and passing in a `WeaData`and `WeaMetaData`objects.

To access a variable during the simulation, you need to first request the variable before running the simulation by calling `request_variable`and passing in a variable name and key. Once the variable is requested, you can access the variable using `get_variable_value` and passing in a variable name and key. 

To set an actuator value during the simulation, use `actuate` and pass in an actuator name and value. Currently, you can only actuate the cfs construction of a window and light electricity rate. To modify the cfs construction, the actuator name is the window name, and the value is the cfs name. To modify the light electricity rate, the actuator name is the light name, and the value is the electricity rate. 


```python
def controller(state):
    nominal_lighting_power = 30

    shade_names = {
        0: "Generic Clear Glass",
        1: "Generic Clear Glass_Satine 5500 5%, White Pearl",
    }

    dt = ep.get_datetime()

    # get variable value that is requested before the simulation run and the callback function is called
    direct_normal_irradiance = ep.get_variable_value(
        name="Site Direct Solar Radiation Rate per Area", key="Environment"
    )
    diffuse_horizontal_irradiance = ep.get_variable_value(
        name="Site Diffuse Solar Radiation Rate per Area", key="Environment"
    )
   
    ## control CFS construction based on time
    if dt.hour > 10 and dt.hour < 15:
        _shades = 1
        tmx = tmx_shaded
    else:
        _shades = 0
        tmx = tmx_unshaded

    ep.actuate("Zn001:Wall001:Win001", ep.handles.construction[shade_names[_shades]])

    ## control lights
    # create WeaData object
    weadata = fr.WeaData(
        time=dt, dni=direct_normal_irradiance, dhi=diffuse_horizontal_irradiance
    )
    # initialize sky/sun matrix
    smx = fr.load_matrix(fr.genskymtx([weadata], ep.wea_meta, mfactor=4))
    # get workplane illuminance
    wpi = fr.multiply_rgb(
        vmx_window1, tmx, dmx_window1, smx, weights=[47.4, 119.9, 11.6]
    )
    avg_wpi = wpi.mean()

    # lighting power, assuming linear dimming curve
    lighting_power = (1 - min(avg_wpi / 500, 1)) * nominal_lighting_power

    ep.actuate("Light_ZONE ONE", lighting_power)
```
### Request output variable
Use `request_output`to request output variables that are not in the input idf file before the simulation run.
```python
epmodel.request_output("Surface Inside Face Solar Radiation Heat Gain Rate per Area")

```
### Initialize pyenergyplus.api to simulate

To access a variable during the simulation, you need to first request the variable before running the simulation by calling `request_variable`and passing in a variable name and key. 

Register the controller functions to be called back by EnergyPlus by calling `set_callback`and passing in a callback point and function. To simulate, use `run`with optional parameters: `-w`weather file, `-d`output directory, and `-p`output prefix (default: eplus).

Refer to [Application Guide for EMS](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v22.1.0/EMSApplicationGuide.pdf) for descriptions of the calling points. This example uses `callback_begin_system_timestep_before_predictor`calling point to control the shading and lighting.

    "The calling point called “BeginTimestepBeforePredictor” occurs near the beginning of each timestep
    but before the predictor executes. “Predictor” refers to the step in EnergyPlus modeling when the
    zone loads are calculated. This calling point is useful for controlling components that affect the
    thermal loads the HVAC systems will then attempt to meet. Programs called from this point
    might actuate internal gains based on current weather or on the results from the previous timestep.
    Demand management routines might use this calling point to reduce lighting or process loads,
    change thermostat settings, etc."


```python
with fr.EnergyPlusSetup(api, epmodel.epjs) as ep:
    ep.request_variable(
        name="Site Direct Solar Radiation Rate per Area", key="Environment"
    )
    ep.request_variable(
        name="Site Diffuse Solar Radiation Rate per Area", key="Environment"
    )
    # create WeaMetaData object
    ep.set_callback("callback_begin_system_timestep_before_predictor", controller)
    ep.run(
        weather_file="USA_CA_Oakland.Intl.AP.724930_TMY3.epw",
        output_prefix="1ZoneUncontrolled_win_1",
    )
```


### Load and visualize results

Use `pd.read_csv`to read the output csv file.

```python
df = pd.read_csv(
    "./1ZoneUncontrolled_win_1out.csv",
    index_col=0,
    parse_dates=True,
    date_parser=fr.ep_datetime_parser,
)
```

Plot data on 07/21


```python
df_0721 = df.loc["1900-07-21"]
```

 From 11:00 to 15:00, the window is shaded, where the fenestration construction is Generic Clear Glass_Satine 5500 5%, White Pearl. Otherwise, the window is unshaded, where the fenestration construction is Generic Clear Glass. The drop in transmitted solar radiation from 11:00 to 15:00 reflects the change in the fenestration state from unshaded to shaded.
 
  The light is linearly dimmed in response to the workplane illuminance. Before sunrise and after sunset, the light is in full power. Then, in the morning from 5:30 to 11:00, when the window is unshaded, the lighting power decreases as the workplane illuminance increases; likewise inversely happened from 15:00 until sunset. From 11:00 to 15:00, the lighting power is higher because the workplane illuminance decreases with the window changed to shaded.


```python
fig, ax = plt.subplots()
ax.plot(
    df_0721[
        "ZN001:WALL001:WIN001:Surface Window Transmitted Solar Radiation Rate [W](TimeStep)"
    ],
    c="k",
    linestyle="-",
    label="ZN001:WALL001:WIN001",
)
ax1 = ax.twinx()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax1.plot(
    df_0721["LIGHT_ZONE ONE:Lights Electricity Rate [W](TimeStep)"],
    c="k",
    linestyle="--",
    label="LIGHT_ZONE ONE (right)",
)

ax.axvspan(
    datetime.datetime.strptime("1900-07-21 11:00", "%Y-%m-%d %H:%M"),
    (datetime.datetime.strptime("1900-07-21 15:00", "%Y-%m-%d %H:%M")),
    color="0.9",
)
ax.annotate(
    "Shaded",
    xy=(datetime.datetime.strptime("1900-07-21 13:00", "%Y-%m-%d %H:%M"), 2250),
    ha="center",
    color="r",
)
ax.annotate(
    "Unshaded",
    xy=(datetime.datetime.strptime("1900-07-21 5:00", "%Y-%m-%d %H:%M"), 2250),
    ha="center",
    color="r",
)
ax.annotate(
    "Unshaded",
    xy=(datetime.datetime.strptime("1900-07-21 21:30", "%Y-%m-%d %H:%M"), 2250),
    ha="center",
    color="r",
)


ax.set(xlabel="Time", ylabel="Window Transmitted Solar Radiation Rate [W]")
ax1.set(ylabel="Lights Electricity Rate [W]")
fig.legend(loc="center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
plt.tight_layout()

```


    
![png](../../assets/output_46_0.png)
    



Plot interior surfaces solar radiation heat gain rate per area


```python
fig, ax = plt.subplots()
y_vals = [
    col
    for col in df_0721.columns
    if "Surface Inside Face Solar Radiation Heat Gain Rate per Area" in col
]

for y_val in y_vals:
    ax.plot(
        df_0721[y_val],
        linestyle="-",
        label=y_val.split(":Surface")[0],
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

ax.set(xlabel="Time", ylabel="Solar Radiation Heat Gain Rate per Area [W/m2]")
fig.legend(loc="center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
plt.tight_layout()

```


    
![png](../../assets/output_47_0.png)
    

### Check implementation

Check if the simulation is implemented correctly. Compare simulation results generated with the controller and no controller. 

Simulate an unshaded single-pane CFS state, not controlled by a controller function.

```python
epmodel.epjs["FenestrationSurface:Detailed"]["Zn001:Wall001:Win001"][
    "construction_name"
] = "Generic Clear Glass"
with fr.EnergyPlusSetup(api, epmodel.epjs) as ep:
    ep.run(
        weather_file="USA_CA_Oakland.Intl.AP.724930_TMY3.epw",
        output_prefix="single_glass",
    )
```




```python
df = pd.read_csv(
    "./single_glassout.csv",
    index_col=0,
    parse_dates=True,
    date_parser=fr.ep_datetime_parser,
)
df_0721 = df.loc["1900-07-21"]

fig, ax = plt.subplots()
y_vals = [
    col
    for col in df_0721.columns
    if "Surface Inside Face Solar Radiation Heat Gain Rate per Area" in col
]

for y_val in y_vals:
    ax.plot(
        df_0721[y_val],
        linestyle="-",
        label=y_val.split(":Surface")[0],
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

ax.set(xlabel="Time", ylabel="Solar Radiation Heat Gain Rate per Area [W/m2]")
fig.legend(loc="center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
plt.tight_layout()
```



![image](../../assets/output_49_0.png)
    


## Example 2 - electrochromic glass with 4 tinted states

### Add 4 tinted electrochromic states to the EnergyPlus model
Each glazing system consists of one layer of ec glass and one layer of clear glass. 

The gap between the glasses is 10% air and 90% argon. The default gap is air at 0.0127 m thickness. To customize the gap, use the `gaps`attribute of the `GlazingSystem`class. The `gaps`attribute is a list of tuples. Each tuple consists of tuples, where the first item is the gas type and the second item is the gas ratio, and a float for the gap thickness. The default thickness is 0.0127 m. Also, to customize the name of the glazing system, use the `name`attribute of the `GlazingSystem`class.


```python
gs_ec01 = fr.GlazingSystem()
gs_ec01.add_glazing_layer(
    "products/igsdb_product_7405.json"
)  # SageGlass SR2.0_7mm lami fully tinted 1%T
gs_ec01.add_glazing_layer("products/CLEAR_3.DAT")
gs_ec01.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
gs_ec01.name = "ec01"

gs_ec06 = fr.GlazingSystem()
gs_ec06.add_glazing_layer(
    "products/igsdb_product_7407.json"
)  # SageGlass SR2.0_7mm lami int state 6%T
gs_ec06.add_glazing_layer("products/CLEAR_3.DAT")
gs_ec06.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
gs_ec06.name = "ec06"

gs_ec18 = fr.GlazingSystem()
gs_ec18.add_glazing_layer(
    "products/igsdb_product_7404.json"
)  # SageGlass SR2.0_7mm lami int state 18%T
gs_ec18.add_glazing_layer("products/CLEAR_3.DAT")
gs_ec18.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
gs_ec18.name = "ec18"

gs_ec60 = fr.GlazingSystem()
gs_ec60.add_glazing_layer(
    "products/igsdb_product_7406.json"
)  # SageGlass SR2.0_7mm lami full clear 60%T
gs_ec60.add_glazing_layer("products/CLEAR_3.DAT")
gs_ec60.gaps = [((fr.AIR, 0.1), (fr.ARGON, 0.9), 0.0127)]
gs_ec60.name = "ec60"
```


```python
gs_ec01.compute_solar_photopic_results()
gs_ec06.compute_solar_photopic_results()
gs_ec18.compute_solar_photopic_results()
gs_ec60.compute_solar_photopic_results()

```


```python
epmodel.add_cfs(gs_ec01)
epmodel.add_cfs(gs_ec06)
epmodel.add_cfs(gs_ec18)
epmodel.add_cfs(gs_ec60)

```
### Define controller function for the EC states
The electrochromic glasses are controlled by time for this demonstration.

```python
def ec_controller(state):
    shade_names = {
        0: "ec01",
        1: "ec06",
        2: "ec18",
        3: "ec60",
    }

    dt = ep.get_datetime()

    ## control CFS construction based on dni
    # get variable value that is requested before the simulation run and the callback function is called
    dni = ep.get_variable_value(
        name="Site Direct Solar Radiation Rate per Area", key="Environment"
    )

    if dni >= 800:
        _shades = 0
    elif dni < 800 and dni >= 600:
        _shades = 1
    elif dni < 600 and dni >= 400:
        _shades = 2
    else:
        _shades = 3

    ep.actuate("Zn001:Wall001:Win001", ep.handles.construction[shade_names[_shades]])
```
### Initialize pyenergyplus.api to simulate
```python
epmodel.request_output("Site Direct Solar Radiation Rate per Area")
epmodel.request_output("Window Transmitted Solar Radiation Rate per Area")
```
```python
with fr.EnergyPlusSetup(api, epmodel.epjs) as ep:
    ep.request_variable(
        name="Site Direct Solar Radiation Rate per Area", key="Environment"
    )

    ep.set_callback("callback_begin_system_timestep_before_predictor", ec_controller)
    ep.run(
        weather_file="USA_CA_Oakland.Intl.AP.724930_TMY3.epw",
        output_prefix="ec",
    )
```
### Load and visualize results

```python
df = pd.read_csv(
    "./ecout.csv",
    index_col=0,
    parse_dates=True,
    date_parser=fr.ep_datetime_parser,
)
df_0721 = df.loc["1900-07-21"]

fig, ax = plt.subplots()
ax.plot(
    df_0721[
        "ZN001:WALL001:WIN001:Surface Window Transmitted Solar Radiation Rate [W](TimeStep)"
    ],
    c="k",
    linestyle="-",
    label="ZN001:WALL001:WIN001",
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

ax1 = ax.twinx()
ax1.plot(df_0721["Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)"], c="r", label="DNI")
ax1.set_ylabel("DNI [W/m2]")
ax.axvspan(
    datetime.datetime.strptime("1900-07-21 9:00", "%Y-%m-%d %H:%M"),
    (datetime.datetime.strptime("1900-07-21 15:30", "%Y-%m-%d %H:%M")),
    color="0.8",
)
ax.axvspan(
    datetime.datetime.strptime("1900-07-21 7:00", "%Y-%m-%d %H:%M"),
    (datetime.datetime.strptime("1900-07-21 9:00", "%Y-%m-%d %H:%M")),
    color="0.85",
)
ax.axvspan(
    datetime.datetime.strptime("1900-07-21 15:30", "%Y-%m-%d %H:%M"),
    (datetime.datetime.strptime("1900-07-21 17:45", "%Y-%m-%d %H:%M")),
    color="0.85",
)

ax.axvspan(
    datetime.datetime.strptime("1900-07-21 06:15", "%Y-%m-%d %H:%M"),
    (datetime.datetime.strptime("1900-07-21 7:00", "%Y-%m-%d %H:%M")),
    color="0.93",
)

ax.axvspan(
    datetime.datetime.strptime("1900-07-21 17:45", "%Y-%m-%d %H:%M"),
    (datetime.datetime.strptime("1900-07-21 18:15", "%Y-%m-%d %H:%M")),
    color="0.93",
)

ax1.annotate(
    "EC 60%",
    xy=(datetime.datetime.strptime("1900-07-21 3:00", "%Y-%m-%d %H:%M"), 900),
    ha="center",
    color="r",
)

ax1.annotate(
    "18%",
    xy=(datetime.datetime.strptime("1900-07-21 06:00", "%Y-%m-%d %H:%M"), 900),
    ha="center",
    color="r",
)

ax1.annotate(
    "6%",
    xy=(datetime.datetime.strptime("1900-07-21 8:00", "%Y-%m-%d %H:%M"), 900),
    ha="center",
    color="r",
)

ax1.annotate(
    "1%",
    xy=(datetime.datetime.strptime("1900-07-21 12:00", "%Y-%m-%d %H:%M"), 900),
    ha="center",
    color="r",
)

ax1.annotate(
    "6%",
    xy=(datetime.datetime.strptime("1900-07-21 16:30", "%Y-%m-%d %H:%M"), 900),
    ha="center",
    color="r",
)

ax1.annotate(
    "18%",
    xy=(datetime.datetime.strptime("1900-07-21 18:30", "%Y-%m-%d %H:%M"), 900),
    ha="center",
    color="r",
)

ax1.annotate(
    "60%",
    xy=(datetime.datetime.strptime("1900-07-21 21:00", "%Y-%m-%d %H:%M"), 900),
    ha="center",
    color="r",
)

ax1.set_ylim(top=1000)
ax.set(xlabel="Time", ylabel="Window Transmitted Solar Radiation Rate [W]")
plt.tight_layout()
```


    
![png](../../assets/output_58_0.png)
    



```python

```
