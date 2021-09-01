![Install + Test](https://github.com/LBNL-ETA/frads/actions/workflows/main.yml/badge.svg)
![CodeQL](https://github.com/LBNL-ETA/frads/actions/workflows/codeql-analysis.yml/badge.svg)
[![Upload Python Package](https://github.com/LBNL-ETA/frads/actions/workflows/python-publish.yml/badge.svg)](https://github.com/LBNL-ETA/frads/actions/workflows/python-publish.yml)
# _frads_: Framework for Radiance matrix-based simulation control (WIP)

This is the repository for _frads_ development. Radiance is a free and open-source, raytracing-based lighting engine that is used extensively by engineering firms for innovative solar control, lighting, and daylighting design to improve the energy efficiency of buildings. With matrix algebraic methods, climate-based annual simulations can now be conducted in less than two minutes. _frads_ automates setup of these simulations by providing end users with an open-source, high-level abstraction of the Radiance command-line workflow (Unix toolbox model), helping to reduce the steep learning curve and associated user errors. _frads_ also provides the necessary infrastructure needed for seamless integration of Radiance and other modeling tools, such as EnergyPlus.

[Documentation](https://frads.readthedocs.io/en/latest/)

## Contact/ Support
We welcome beta users of _frads_. Feel free to post questions and suggestions in the Discussion section of this GitHub site or contact the principal author at taoningwang@lbl.gov.
Information about Radiance can be found at: https://www.radiance-online.org .
The Radiance community is active and welcomes new users via the Radiance Discourse site or Unmet Hours.

## Testing
_frads_ uses Radiance tools in its implementation. Radiance models have been rigorously tested and validated using laboratory and outdoor field data, demonstrating its superior  performance in delivering photometrically accurate, photorealistic results. Each Radiance commit and release is tested using the GitHub Action system.  Unit tests were developed for most of the major Radiance programs. Tests are performed using Radiance _radcompare_, which was designed specifically to test Monte Carlo ray-tracing algorithms.
Integration tests are the main type of test performed for _frads_ commit and releases.  These tests also use the GitHub Action system.

## Releases
_frads_ is a work in progress (see to-do list below). _frads_ has been tested on the latest official release of Radiance (September 2020, v5.3) but may not have been tested on the latest HEAD release, which contains source code changes made as recently as yesterday. _frads_ has also been tested on the latest official EnergyPlus release (> v9.3).

## Installation

_frads_ runs from the terminal prompt (command line) on Windows, Mac, and Linux OS. Radiance must be [installed](https://www.radiance-online.org/download-install/radiance-source-code/latest-release) prior to use of _frads_.  You can then install _frads_ by entering the following command in your terminal/cmd/powershell:

```
pip install frads
```

You can also install _frads_ from this Github repository using this command:

```
pip install git+https://github.com/LBNL-ETA/frads.git
```

## To do
- [x] 2-, 3-, 4-, and 5-phase methods implemented (image-based 5-phase works on Windows by aggressive sun-culling, 4-phase to be tested)
- [x] epJSON to Radiance workflow implemented (not fully tested), window puncher + wall thickener
- [x] EnergyPlus Radiance runtime interaction preliminarily implemented
- [ ] Automated window subdivision analysis
- [ ] Implement daylight metrics calculation with EnergyPlus integration
- [ ] Implement thermal and visual comfort calculation with EnergyPlus integration
- [ ] Link to global fenestration systems database and implement BSDF combination routines
- [ ] Spawn of EnergyPlus integration, variable timestep detailed HVAC and control modeling

## Reference

Wang, T., "Frads: A Python Library for Radiance Simulation Control", 2021 Radiance workshop, Bilbao, Spain, August 19, 2021, [ppt](https://www.radiance-online.org/community/workshops/2021-bilbao-spain-2/presentations/19_thursday/frads.pdf) , [voice recording](https://www.radiance-online.org/community/workshops/2021-bilbao-spain-2)

Wang, T., Ward, G., and Lee, E.S. (2021), A Python Library for Radiance Matrix-based Simulation Control and EnergyPlus Integration, Proceedings of Building Simulation 2021, International Building Performance Simulation Association, Bruges, September 1-3, 2021. Publication to be posted: [pdf](https://eta.lbl.gov/publications)

