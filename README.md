![Install + Test](https://github.com/LBNL-ETA/frads/actions/workflows/main.yml/badge.svg)
![CodeQL](https://github.com/LBNL-ETA/frads/actions/workflows/codeql-analysis.yml/badge.svg)
# Radiance matrix-based simulation control (WIP)

_frads_ is a open-source python-based high-level abstraction of Radiance command-line workflow for multi-phase matrix-based simulation and beyond. It consists of

1) a series of commandline tools that automates standard workflow, and;

2) a library that facilitates Radiance matrix-based simulation workflow setup.

It also support EnergyPlus(>=9.3) integration with its new EMS module which allows EnergyPlus to be used as a library.

[Documentation](https://frads.readthedocs.io/en/latest/)

## Installation

You can install _frads_ by entering the following command in your terminal/cmd/powershell:

```
pip install git+https://github.com/LBNL-ETA/frads.git
```

## To do
- [x] 2,3,4,5-phase methods implemented, (image based 5-phase not working on windows, 4-phase to be tested)
- [x] epJSON to Radiance workflow implemented (not fully tested), window puncher + wall thickner
- [x] EnergyPlus Radiance runtime interaction preliminarily implemented
- [ ] Automated window subdivision analysis
- [ ] Implement daylight metrics calculation with EnergyPlus integration
- [ ] Implement thermal and visual comfort calculation with EnergyPlus integration
- [ ] Link to global fenestration systems database and implement BSDF combination routines
- [ ] Spawn of EnergyPlus integration, variable timestep detailed HVAC and control modeling

## Reference
