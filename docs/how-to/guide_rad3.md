# How to calculate workplane illuminance and edgps using three-phase method?

This guide will show you how to calculate workplane illuminance and eDGPs (enhanced simplified Daylight Glare Probability) using the Three-Phase method in Radiance.

**What is the Three-Phase method?**

The Three-Phase method a way to perform annual daylight simulation of complex fenestration systems. The method divide flux transfer into three phases or matrices:

* V(iew): flux transferred from simulated space to the interior of the fenestration
* T(ransmission): flux transferred through the fenestration (usually represented by a BSDF)
* D(aylight): flux transferred from the exterior of fenestration to the sky

Multiplication of the three matrices with the sky matrix gives the illuminance at the simulated point. In the case where one wants to calculate the illuminance for different fenestration systems, one only needs to calculate the daylight and view matrice once and then multiply them with the transmission matrix of each fenestration system.


**Workflow for setting up a three-phase method**

1. Initialize a ThreePhaseMethod instance with a workflow configuration.

2. (Optional) Save the matrices to file. 

3. Generate matrices.

4. Calculate workplane illuminance and eDGPs.

## 0. Import the required classes and functions

```python
from datetime import datetime
import frads as fr
```

## 1. Initialize a ThreePhaseMethod instance with a workflow configuration

To set up a Three-Phase method workflow, call the `ThreePhaseMethod` class and pass in a workflow configuration that contains information about the settings and model. See [How to set up a workflow configuration?](guide_rad2.md/) for more information.


??? example "cfg"
    ```json
    dict1 = {
        "settings": {
            "method": "3phase",
            "sky_basis": "r1",
            "epw_file": "",
            "wea_file": "oak.wea",
            "sensor_sky_matrix": ["-ab", "0"],
            "view_sky_matrix": ["-ab", "0"],
            "sensor_window_matrix": ["-ab", "0"],
            "view_window_matrix": ["-ab", "0"],
            "daylight_matrix": ["-ab", "0"],
        },
        "model": {
            "scene": {
                "files": ["walls.rad", "ceiling.rad", "floor.rad", "ground.rad"]
            },
            "windows": {
                "window1": {
                    "file": "window1.rad",
                    "matrix_name": "window1_matrix",
                }
            },
            "materials": {
                "files": ["materials.mat"],
                "matrices": {"window1_matrix": {"matrix_file": "window1_bsdf.xml"}},
            },
            "sensors": {
                "sensor1": {"file": "sensor1.txt"},
                "view1": {"data": [[1, 1, 1, 0, -1, 0]]},
            },
            "views": {"view1": {"file": "view1.vf"}},
        },
    }

    ```
    
    ``` python
    cfg = fr.WorkflowConfig.from_dict(dict1)
    ```

```python
workflow = fr.ThreePhaseMethod(cfg) 
```

## 2. (Optional) Save the matrices to file

A *.npz file will be generated in the current working directory. The file name is a hash string of the configuration content.

```python
workflow.config.settings.save_matrices = True # default=False
```

## 3. Generate matrices
Use the `generate_matrices()` method to generate the following matrices:

- View --> window
- Sensor --> window
- Daylight

```python
workflow.generate_matrices()
```

!!! tip "get workflow from EnergyPlusSetup"
    If you are using the ThreePhaseMethod class in EnergyPlusSetup, you can get the workflow from the EnergyPlusSetup instance. See [How to enable Radiance in EnergyPlus simulation?](guide_radep1.md) for more information.
    
    ```python
    eps = fr.EnergyPlusSetup(epmodel, weather_file, enable_radiance=True)
    workflow = eps.rworkflows[zone_name]
    ```

## 4. Calculate

### 4.1 workplane illuminance

Use the `calculate_sensor()` method to calculate workplane illuminance for a sensor. Need to pass in the name of the sensor, a dictionary of window names and their corresponding BSDF matrix file names, datetime, direct normal irradiance (DNI), and diffuse horizontal irradiance (DHI).

```python
workflow.calculate_sensor(
    sensor="sensor1",
    bsdf={"window1": "window1_matrix"},
    time=datetime(2023, 1, 1, 12),
    dni=800,
    dhi=100,
)
```

**what does calculate_sensor() do behind the scene?**

It multiplies the view, transmission, daylight, and sky matrices
with weights in the red, green, and blue channels to get the illuminance at the sensor point.

### 4.2 eDGPs

**What is eDGPs?**

eDGPs is an enhanced version of the simplified Daylight Glare Probability (DGPs) to evaluate the glare potential.

Use the `calculate_edgps()` method to calculate eDGPs for a view. Need to pass in the name of the view, a dictionary of window names and their corresponding BSDF matrix file names, datetime, direct normal irradiance (DNI), diffuse horizontal irradiance (DHI), and ambient bounce.

!!! Note
    To calculate eDGPs for view1, you need to specify a view1 key name in `dict1["model"]["views"]` and `dict1["model"]["sensors"]`.

```python
workflow.calculate_edgps(
    view="view1", 
    bsdf={"window1": "window1_matrix"},
    time=datetime(2023, 1, 1, 12),
    dni=800,
    dhi=100,
    ambient_bounce=1,
)
```

**what does calculate_edgps() do behind the scene?**

First, use Radiance `rpict` to render a low-resolution (ambient bouce: 1) high dynamic range image (HDRI). Then, use Radiance `evalglare` to evaluate the DGP of the HDRI with illuminance modification calculated using the matrix multiplication.
