# How to use the Three-Phase method to calculate eDGPs and workplane illuminance?

This guide will show you how to set up a Three-Phase method workflow in Radiance to calculate eDGPs and workplane illuminance. 

To set up a Three-Phase method workflow, call the `ThreePhaseMethod` class by passing in a workflow configuration that contains information about the settings and model. See [How to set up a workflow configuration?](../guide7/) for more information.


**Workflow for setting up a three-phase method**

1. Initialize a ThreePhaseMethod instance with a workflow configuration.

2. (Optional) Save the matrices to file. 

3. Generate matrices.

4. Calculate eDGPs or workplane illuminance.

## 0. Import the required classes and functions

```python
from datetime import datetime
import frads as fr
```

## 1. Initialize a ThreePhaseMethod instance with a workflow configuration


??? example "cfg"
    ```json
    dict_1 = {
        "settings": {
            "method": "3phase",
            "sky_basis": "r1",
            "epw_file": "",
            "wea_file": "oak.wea",
            "sensor_sky_matrix": ["-ab", "0"],
            "view_sky_matrix": ["-ab", "0"],
            "sensor_window_matrix": ["-ab", "0"],
            "view_window_matrix": ["-ab", "0"],
            "daylight_matrix": ["-ab", "0"]
        },
        "model": {
            "scene": {
                "files": [
                    "walls.rad",
                    "ceiling.rad",
                    "floor.rad",
                    "ground.rad"
                ]
            },
            "windows": {
                "window_1": {
                    "file": "window_1.rad",
                    "matrix_file": "window_1.xml"
                }
            },
            "materials": {
                "files": ["materials.mat"]
            },
            "sensors": {
                "sensor_1": {"file": "sensor_1.txt"},
            },
            "views": {
                "view_1": {
                    "file": "view_1.vf", 
                    "xres": 16,
                    "yres": 16
                }
            }
        }
    }
    ```
    
    ``` python
    cfg = fr.WorkflowConfig.from_dict(dict_1)
    ```

```python
workflow = fr.ThreePhaseMethod(cfg) 
```

## 2. (Optional) Save the matrices to file

A *.npz file will be generated in the current working directory. The file name is a hash string of the configuration content.

```python
workflow.config.settings.save_matrices = True #default=False
```

## 3. Generate matrices
Use the `generate_matrices()` method to generate the following matrices:

- View --> window
- Sensor --> window
- Daylight

```python
workflow.generate_matrices()
```

## 4.1 Calculate eDGPs

```python
workflow.calculate_edgps(
    view="view_1",
    shades=["window_1.rad"], #shade geometry files
    bsdf=workflow.window_bsdfs["window_1"], #shade BSDF
    date_time=datetime(2023, 1, 1, 12),
    dni=800,
    dhi=100,
    ambient_bounce=1,
)
```

## 4.2 Calculate workplane illuminance

```python
workflow.calculate_sensor(
    sensor="sensor_1",
    bsdf=rad_workflow.window_bsdfs["window_1"], # shade BSDF
    date_time=datetime(2023, 1, 1, 12),
    dni=800,
    dhi=100,
)
```