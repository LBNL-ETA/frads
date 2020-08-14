# Radiance matrix-based simulation control (WIP)

frads is a python3-based higher-level abstraction of Radiance command-line workflow for multi-phase matrix-based simulation and beyond. It consists of 

1) a series of commandline tools, in the bin directory, that automates standard Radiance workflow, and;

2) a library that facilitates Radiance matrix-based simulation workflow setup.

It also support EnergyPlus(>=9.3) integration with its new EMS module which allows EnergyPlus to be used as a library.

## To do
- [x] 2,3,4,5-phase methods implemented, (image based 5-phase not working on windows, 4-phase to be tested)
- [ ] Automated window subdivision analysis
- [x] epJSON to Radiance workflow implemented (not fully tested), window puncher + wall thickner
- [x] EnergyPlus Radiance runtime interaction preliminarily implemented
- [ ] Implement daylight metrics calculation with EnergyPlus integration
- [ ] Implement thermal and visual comfort calculation with EnergyPlus integration
- [ ] Link to global fenestration systems database and implement BSDF combination routines
- [ ] Spawn of EnergyPlus integration, variable timestep detailed HVAC and control modeling

## Installation

You can install stable frads by entering the following command in your terminal/cmd/powershell:

```
pip install frads
```
If you need the latest frads you can clone this repository, or download and unzip the pacakge, and use the command:
```
pip install /PATH/TO/YOUR/frads
```


## EnergyPlus integration
With the advent of EnergyPlus 9.3, EnergyPlus core engine is semi-exposed through its Energy Management System through C/Python application programming interface.

frads leverage this new feature to integrate Radiance simulation engine with EnergyPlus by manipulating EnergyPlus EMS 'Actuators' at run-time. For a standard workflow, an user uses an EnergyPlus model (epJSON) as the input, specifies the one or more standard operation (e.g. sDA calculation using Radiance matrix-based methods.)
### Usage
An user can use the commandline tool, eprad, to carry out a Radiance/EnergyPlus simulation
```
eprad.py test.epJSON (options)
```
One can also use the library by importing neede function to facilitate a specific workflow
```python
import epjson2rad

radobj = epjson2rad.epJSON2Rad(epsj_path)
```
