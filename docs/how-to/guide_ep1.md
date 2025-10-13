# How to run a simple EnergyPlus simulation?

This guide will show you how to run a simple EnergyPlus simulation. After loading an EnergyPlus model, you can edit the objects and parameters in the model before running the simulation.


## 0. Import required Python libraries

```python
import frads as fr
```
**Optional: Load reference EnergyPlus model and weather files**

You will need a working EnergyPlus model in idf or epjson format to initialize an EnergyPlus model. Or you can load an EnergyPlus reference model from `pyenergyplus.dataset`.

```python
from pyenergyplus.dataset import ref_models, weather_files
```

!!! tip "Tips: Reference EnergyPlus models and weather files"
    The `pyenergyplus.dataset` module contains a dictionary of EnergyPlus models and weather files. The keys are the names of the models and weather files. The values are the file paths to the models and weather files.

    ```python
    ref_models.keys()
    ```

    > ```
    dict_keys([
        'full_service_restaurant', 'hospital', 'large_hotel', 
        'large_office', 'medium_office', 'midrise_apartment', 
        'outpatient', 'primary_school', 'quick_service_restaurant',
        'secondary_school', 'small_hotel', 'small_office', 
        'standalone_retail', 'strip_mall', 'supermarket', 'warehouse'
    ])
    > ```
    
## 1 Initialize an EnergyPlus model

Initialize an EnergyPlus model by calling `load_energyplus_model` and passing in a working idf or epjson file path.


### 1.1 Define EnergyPlus model file path

=== "local file"
    ``` python
    idf = "medium_office.idf"
    ```

=== "reference model"
    ``` python
    idf = ref_models["medium_office"] # (1) 
    ```

    1.  from pyenergyplus.dataset

### 1.2 Load the EnergyPlus model

```python
epmodel = fr.load_energyplus_model(idf)
```

## 2 Edit the EnergyPlus model (optional)

### All EnergyPlus objects

You can access any EnergyPlus model objects (simulation parameters) as you would do to a class attribute. The EnergyPlus model objects share the same name as that in the Input Data File (IDF) but in lower case separated by underscores. For example, the `FenestrationSurface:Detailed` object in IDF is `fenestration_surface_detailed` in `EnergyPlusModel`.

```python
epmodel.fenestration_surface_detailed
```
> ```
{'Perimeter_bot_ZN_1_Wall_South_Window': FenestrationSurfaceDetailed(surface_type=<SurfaceType1.window: 'Window'>, construction_name='Window Non-res Fixed', building_surface_name='Perimeter_bot_ZN_1_Wall_South', outside_boundary_condition_object=None, view_factor_to_ground=<CeilingHeightEnum.autocalculate: 'Autocalculate'>, frame_and_divider_name=None, multiplier=1.0, number_of_vertices=NumberOfVertice2(root=4.0), vertex_1_x_coordinate=1.5, vertex_1_y_coordinate=0.0, vertex_1_z_coordinate=2.3293, vertex_2_x_coordinate=1.5, vertex_2_y_coordinate=0.0, vertex_2_z_coordinate=1.0213, vertex_3_x_coordinate=10.5, vertex_3_y_coordinate=0.0, vertex_3_z_coordinate=1.0213, vertex_4_x_coordinate=10.5, vertex_4_y_coordinate=0.0, vertex_4_z_coordinate=2.3293)}'
>```

!!! example "Example: Edit the `fenestration_surface_detailed` object"

    ```python title="Change the construction name of the window"
    epmodel.fenestration_surface_detailed[
        "Perimeter_bot_ZN_1_Wall_South_Window"
    ].construction_name = "gs1"
    ```

!!! example "Example: Edit the `lights` object"

    ```python title="Change the watts per zone floor area"
    epmodel.lights["Perimeter_bot_ZN_1_Lights"].watts_per_zone_floor_area = 10
    ```


### Glazing system (complex fenestration system)

Use `EnergyPlusModel.add_glazing_system()` to easily add glazing system (complex fenestration systems) to the `construction_complex_fenestration_state` object in the EnergyPlus model.

First, use the `create_glazing_system()` function to create a glazing system. Then use `EnergyPlusModel.add_glazing_system()` to add the glazing system to the EnergyPlus. See [How to create a glazing system?](guide_ep2.md) for more details.

``` python
epmodel.add_glazing_system(gs1) # (1) 
```

1.  `gs1 = fr.create_glazing_system(name="gs1", layer_inputs=[fr.LayerInput("product1.json"), fr.LayerInput("product2.json")])`
    
### Lighting

Use `EnergyPlusModel.add_lighting()` to easily add lighting systems to the `lights` object in the EnergyPlus model. The function takes in the name of the zone to add lighting, the lighting level in the zone in Watts, and an optional `replace` argument. If `replace` is `True`, the zone's existing lighting system will be replaced by the new lighting system. If `replace` is `False` and the zone already has a lighting system, an error will be raised. The default value of `replace` is `False`.

```python
epmodel.add_lighting(zone="Perimeter_bot_ZN_1", lighting_level=10, replace=True)
```

### Add Output 

Use `EnergyPlusModel.add_output()` to easily add output variables or meters to the `Output:Variable` or `Output:Meter` object in the EnergyPlus model. The  method takes in the type of the output (variable or meter), name of the output, and the reporting frequency. The default reporting frequency is `Timestamp`. 

```python
epmodel.add_output(
    output_type="variable",
    output_name="Lights Electricity Rate",
    reporting_frequency="Hourly",
)
```

!!! Tip 
    See .rdd file for all available output variables and .mdd file for all available output meters.


## 3. Run the EnergyPlus simulation

Call `EnergyPlusSetup` class to set up the EnergyPlus simulation. `EnergyPlusSetup` takes in the EnergyPlus model and an optional weather file. If no weather file is provided, when calling `EnergyPlusSetup.run()`, you need to set `design_day` to `True` and run design-day-only simulation; otherwise, an error will be raised. Annual simulation requires a weather file.

### 3.1 Define weather file path (optional)

=== "local file"
    ```python
    weather_file = "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
    ```

=== "reference weather file"
    ```python
    weather_file = weather_files["usa_ca_san_francisco"] # (1) 
    ```

    1.  from pyenergyplus.dataset


### 3.2 Initialize EnergyPlus simulation setup
```python 
epsetup = fr.EnergyPlusSetup(epmodel, weather_file)
```

### 3.3 Run the EnergyPlus simulation
Call `EnergyPlusSetup.run()` to run the EnergyPlus simulation. This will generate EnergyPlus output files in the working directory. 

The function has the following arguments:

* output_directory: Output directory path. (default: current directory)
* output_prefix: Prefix for output files. (default: eplus)
* output_suffix: Suffix style for output files. (default: L)
*     L: Legacy (e.g., eplustbl.csv)
*     C: Capital (e.g., eplusTable.csv)
*     D: Dash (e.g., eplus-table.csv)
* silent: If True, do not print EnergyPlus output to console. (default: False)
* annual: If True, force run annual simulation. (default: False)
* design_day: If True, force run design-day-only simulation. (default: False)

=== "simple"
    ```python
    epsetup.run()
    ```

=== "annual"
    ```python
    # need a weather file
    epsetup.run(annual=True)
    ```

=== "design day"
    ```python
    # need to set up design day parameters in EnergyPlus model.
    epsetup.run(design_day=True)
    ```

 


