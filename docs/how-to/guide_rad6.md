# How to calculate melanopic illuminance?

This guide will show you how to calculate melanopic illuminance (mev) using the Three-Phase method in Radiance with EnergyPlus integration. The guide demonstrates both photopic illuminance and melanopic illuminance calculations.

**What is melanopic illuminance ?**

Melanopic illuminance (mev), also known as melanopic lux, is a measure of light's effectiveness at stimulating the melanopsin-containing intrinsically photosensitive retinal ganglion cells (ipRGCs) in the human eye. These cells are responsible for non-visual responses to light, including:

* Circadian rhythm regulation
* Melatonin suppression
* Alertness and sleep-wake cycle control
* Pupillary light reflex

Unlike photopic illuminance, which measures light based on the photopic luminous efficiency function that peaks at 555 nm (green light), melanopic illuminance is weighted towards shorter wavelengths, with peak sensitivity around 490 nm (blue-cyan light). This makes melanopic lux particularly relevant for evaluating lighting conditions for circadian health and alertness in buildings.

**Workflow**

1. [Setup an EnergyPlus Model](#1-setup-an-energyplus-model)

2. [Create glazing systems with melanopic BSDF](#2-create-glazing-systems-with-melanopic-bsdf)

3. [Setup EnergyPlus Simulation with Radiance](#3-setup-energyplus-simulation-with-radiance)

4. [Add melanopic BSDF to the simulation](#4-add-melanopic-bsdf-to-the-simulation)

5. [Calculate photopic and melanopic illuminance](#5-calculate-photopic-and-melanopic-illuminance)

## 0. Import required Python libraries

```python
import frads as fr
from pyenergyplus.dataset import ref_models, weather_files
from datetime import datetime
```

## 1. Setup an EnergyPlus Model

You will need a working EnergyPlus model in idf or epjson format to initialize an EnergyPlus model. Or you can load an EnergyPlus reference model from `pyenergyplus.dataset`. See [How to run a simple EnergyPlus simulation?](guide_ep1.md) for more information on how to setup an EnergyPlus model.

```python
epmodel = fr.load_energyplus_model(ref_models["medium_office"])
```

## 2. Create glazing systems with melanopic BSDF

To calculate melanopic illuminance, you need to create glazing systems with melanopic BSDF data by setting `mbsdf=True` when calling `create_glazing_system()`. This additional step generates the spectral data needed for melanopic calculations.

!!! note "Key difference from standard workflow"
    The main difference compared to standard photopic illuminance calculation is setting `mbsdf=True` when creating glazing systems. This generates additional melanopic BSDF matrices for each glazing system.

### Example 1: Double glazing system

Create a double-pane glazing system with electrochromic glass:

```python
gs2 = fr.create_glazing_system(
    name="gs1",
    layer_inputs=[
        fr.LayerInput("igsdb_product_7406.json"),  # electrochromic glass
        fr.LayerInput("CLEAR_3.DAT"),  # clear glass
    ],
    nproc=4,
    mbsdf=True,  # (1)
)
```

1. Enable melanopic BSDF generation for both layers in the glazing system.

### Add glazing systems to EnergyPlus model

```python
epmodel.add_glazing_system(gs1)
```

## 3. Setup EnergyPlus Simulation with Radiance

Initialize EnergyPlus simulation setup by calling `EnergyPlusSetup` with `enable_radiance=True`. See [How to enable Radiance in EnergyPlus simulation?](guide_radep1.md) for more information.

```python
epsetup = fr.EnergyPlusSetup(
    epmodel,
    weather_files["usa_ca_san_francisco"],
    enable_radiance=True,  # (1)
)
```

1. Enable Radiance for daylighting calculations. This is required for both photopic and melanopic illuminance calculations.

## 4. Add melanopic BSDF to the simulation

After creating the glazing systems with melanopic BSDF, you need to add them to the EnergyPlus simulation setup. This registers the melanopic BSDF data for use in calculations.

!!! note "Additional step for melanopic calculations"
    This step is unique to melanopic illuminance calculations. For photopic illuminance, you do not need to call `add_melanopic_bsdf()`.

```python
epsetup.add_melanopic_bsdf(gs1)
```

## 5. Calculate photopic and melanopic illuminance

Now you can calculate both photopic illuminance and melanopic illuminance for a specific zone and time.

### Define calculation parameters

```python
zone = "Perimeter_bot_ZN_1"  # zone name
window = "Perimeter_bot_ZN_1_Wall_South_Window"  # window name
time = datetime(2025, 7, 31, 12, 0)  # date and time
dni = 800  # direct normal irradiance (W/m²)
dhi = 100  # diffuse horizontal irradiance (W/m²)
sky_cover = 0.5  # sky cover fraction (0-1)
```

### Calculate photopic illuminance using calculate_sensor

Calculate photopic illuminance:

```python
ev = epsetup.rworkflows[zone].calculate_sensor(
    sensor=zone,
    bsdf={window: "gs1"},  # (1)
    time=time,
    dni=dni,
    dhi=dhi,
)
```

1. Dictionary mapping window name to glazing system name. Multiple windows can be specified.

!!! tip
    `calculate_sensor()` returns an array of illuminance values for each sensor point. Use `.mean()` to get the average illuminance across all sensor points.

### Calculate melanopic illuminance using calculate_mev

Calculate melanopic illuminance:

```python
mev = epsetup.rworkflows[zone].calculate_mev(
    sensor=zone,
    bsdf={window: "gs1"},  # (1)
    time=time,
    dni=dni,
    dhi=dhi,
    sky_cover=sky_cover,  # (2)
)
```

1. Dictionary mapping window name to glazing system name.
2. Sky cover fraction is required for melanopic calculations to account for spectral variations in sky conditions.

## Summary

Key differences between photopic and melanopic illuminance calculations:

| Aspect | Photopic Illuminance | Melanopic Illuminance |
|--------|---------------------|----------------------|
| Function | `calculate_sensor()` | `calculate_mev()` |
| Glazing system setup | `create_glazing_system()` | `create_glazing_system(mbsdf=True)` |
| Additional setup | None | `epsetup.add_melanopic_bsdf(gs)` |
| Required parameters | zone, window, time, dni, dhi | zone, window, time, dni, dhi, **sky_cover** |
| Units | lux | melanopic lux |
| Purpose | Visual task performance | Circadian and non-visual effects |

!!! tip "When to use melanopic illuminance"
    Calculate melanopic illuminance when evaluating:
    
    * Circadian lighting design
    * Lighting for alertness and productivity
    * Compliance with WELL Building Standard or similar
    * Optimization of window shading for circadian health
    * Comparison of different glazing systems for non-visual effects
