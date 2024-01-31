# How to enable Radiance in EnergyPlus simulation?

This guide will show you how to enable Radiance in EnergyPlus simulation. 

Users can enable Radiance for desired accuracy in daylighting simulation. Radiance can be used to calculate workplane illuminance, eDGPs, and etc. See [How to calculate workplane illuminance and eDGPs using three-phase method?](guide_rad3.md) for more information.

**Workflow**

1. [Setup an EnergyPlus Model](#1-setup-an-energyplus-model)

2. [Setup EnergyPlus Simulation](#2-setup-energyplus-simulation)


## 0. Import the required classes and functions

```python
import frads as fr
```

## 1. Setup an EnergyPlus Model

You will need a working EnergyPlus model in idf or epjson format to initialize an EnergyPlus model. Or you can load an EnergyPlus reference model from `pyenergyplus.dataset`. See [How to run a simple EnergyPlus simulation?](guide_ep1.md) for more information on how to setup an EnergyPlus model.

```python
epmodel = fr.load_energyplus_model("medium_office.idf")
```

## 2. Setup EnergyPlus Simulation
Initialize EnergyPlus simulation setup by calling `EnergyPlusSetup` and passing in an EnergyPlus model and an optional weather file.

To enable Radiance for daylighting simulation, set `enable_radiance` to `True`; default is `False`. When `enable_radiance` is set to `True`, the `EnergyPlusSetup` class will automatically setup the three-phase method in Radiance. 

```python
epsetup = fr.EnergyPlusSetup(
    epmodel, weather_files["usa_ca_san_francisco"], enable_radiance=True
)
```

After the radiance is enabled, the following calculations can be performed:

=== "Workplane illuminance"

    `epsetup.calculate_wpi()` [more info](../ref/eplus.md#frads.EnergyPlusSetup.calculate_wpi)

    or 

    `epsetup.rworkflows[zone_name].calculate_sensor()`[more info](../ref/threephase.md#frads.ThreePhaseMethod.calculate_sensor)

=== "Simplified Daylight Glare Probability (eDGPs)"

    `epsetup.calculate_edgps()` [more info](../ref/eplus.md#frads.EnergyPlusSetup.calculate_edgps)

    or 

    `epsetup.rworkflows[zone_name].calculate_edgps()`[more info](../ref/threephase.md#frads.ThreePhaseMethod.calculate_edgps)


See [How to calculate workplane illuminance and eDGPs using three-phase method?](guide_rad3.md) and [How to simulate spatial daylight autonomy using three-phase method?](guide_radep3.md) for more information on how to calculate workplane illuminance and eDGPs using Radiance.
