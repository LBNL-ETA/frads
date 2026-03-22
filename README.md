# frads: Framework for Radiance and EnergyPlus Simulation

[![Install + Test](https://github.com/LBNL-ETA/frads/actions/workflows/main.yml/badge.svg)](https://github.com/LBNL-ETA/frads/actions/workflows/main.yml)
[![PyPI](https://img.shields.io/pypi/v/frads.svg)](https://pypi.org/project/frads/)
[![Downloads](https://img.shields.io/pypi/dm/frads.svg)](https://pypi.org/project/frads/)
[![Python](https://img.shields.io/pypi/pyversions/frads.svg)](https://pypi.org/project/frads/)

`frads` is a Python library for building lighting and energy simulation. It provides high-level abstractions over [Radiance](https://www.radiance-online.org) and [EnergyPlus](https://energyplus.net), automating matrix-based annual daylight simulation workflows and enabling Radiance–EnergyPlus co-simulation.

## Features

- **Matrix-based daylight simulation** — automates 2-phase, 3-phase, and 5-phase Radiance workflows for fast, accurate annual simulations
- **EnergyPlus co-simulation** — couples Radiance illuminance calculations with EnergyPlus at each timestep via the EnergyPlus Python API
- **Complex fenestration systems** — creates and manages BSDF glazing systems (electrochromic, venetian blinds, fabric shades) using [pyWinCalc](https://github.com/LBNL-ETA/pyWinCalc)
- **Dynamic shading control** — implements occupancy-based daylight dimming, glare control, and thermal pre-cooling in a single simulation loop
- **Sky and weather** — parses EPW/WEA files and generates Perez all-weather and CIE sky models

## Installation

```bash
pip install frads
```

All dependencies, including Radiance (via `pyradiance`) and EnergyPlus (via `pyenergyplus_lbnl`), are installed automatically.

## Quick Start

**Run a Radiance three-phase annual simulation:**

```python
import frads as fr

cfg = fr.WorkflowConfig.from_dict({
    "settings": {
        "method": "3phase",
        "wea_file": "weather.wea",
    },
    "model": {
        "scene": {"files": ["walls.rad", "floor.rad", "ceiling.rad"]},
        "windows": {"window1": {"file": "window.rad", "matrix_name": "bsdf1"}},
        "materials": {
            "files": ["materials.mat"],
            "matrices": {"bsdf1": {"matrix_file": "window.xml"}},
        },
        "sensors": {"workplane": {"file": "grid.txt"}},
    },
})

workflow = fr.ThreePhaseMethod(cfg)
workflow.generate_matrices()
illuminance = workflow.calculate_sensor(
    sensor="workplane",
    bsdf={"window1": "bsdf1"},
    time=..., dni=800, dhi=100,
)
```

**Run an EnergyPlus simulation with Radiance daylighting:**

```python
import frads as fr

epmodel = fr.load_energyplus_model("office.idf")
epmodel.add_glazing_system(
    fr.create_glazing_system("clear", [fr.LayerInput("clear.json")])
)

with fr.EnergyPlusSetup(epmodel, "weather.epw", enable_radiance=True) as eps:
    def controller(state):
        if not eps.api.exchange.api_data_fully_ready(state):
            return
        wpi = eps.calculate_wpi(zone="Zone1", cfs_name={"Window1": "clear"})
        eps.actuate_lighting_power("Zone1_Lights", (1 - min(wpi.mean() / 500, 1)) * 1000)

    eps.set_callback("callback_begin_system_timestep_before_predictor", controller)
    eps.run(annual=True)
```

## Documentation

Full documentation, how-to guides, and API reference are at **[lbnl-eta.github.io/frads](https://lbnl-eta.github.io/frads/)**.

## Citation

Wang, T., Ward, G., and Lee, E.S. (2021). A Python Library for Radiance Matrix-based Simulation Control and EnergyPlus Integration. *Proceedings of Building Simulation 2021*, IBPSA, Bruges. [PDF](https://www.researchgate.net/publication/358969936_A_Python_Library_for_Radiance_Matrix-based_Simulation_Control_and_EnergyPlus_Integration)

## License

Framework for Radiance Simulation Control (frads) Copyright (c) 2019, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved. See [license.txt](license.txt) for details.
