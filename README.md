![Install + Test](https://github.com/LBNL-ETA/frads/actions/workflows/main.yml/badge.svg)
[![Upload Python Package](https://github.com/LBNL-ETA/frads/actions/workflows/python-publish.yml/badge.svg)](https://github.com/LBNL-ETA/frads/actions/workflows/python-publish.yml)
![Downloads](https://img.shields.io/pypi/dm/frads.svg)
# _frads_: Framework for lighting and energy simulation

This is the repository for _frads_ development. _frads_ faciliates lighting and energy simulation by calling Radiance and EnergyPlus
within the Python environment. Radiance is a free and open-source, raytracing-based lighting engine that is used extensively
by engineering firms for innovative solar control, lighting, and daylighting design to improve the energy efficiency of buildings.
With matrix algebraic methods, climate-based annual simulations can now be conducted in less than two minutes. _frads_ automates setup
of these simulations by providing end users with an open-source, high-level abstraction of the Radiance command-line workflow (Unix toolbox model),
helping to reduce the steep learning curve and associated user errors. _frads_ also provides the necessary infrastructure needed for seamless
integration of Radiance and other modeling tools, such as EnergyPlus.

## [Documentation](https://lbnl-eta.github.io/frads/)

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

Before you can use frads, you need to install it.

### Install Python

Being a Python based library, you'll need to install Python first.
Python version **3.8** or newer is required for frads.

Get the latest version of Python at https://www.python.org/downloads/ or with your operating system’s package manager.

You can verify that Python is installed by typing python from your cmd/powershell/terminal; you should see something like:

	$ python
	Python 3.X.X
	[GCC 4.x] on linux
	Type "help", "copyright", "credits" or "license" for more information.
	>>>

After you have Python installed, you should have `pip` command available in your shell environment as well. You can then use `pip` to install `frads`:

### Install frads

After you have pyenergyplus installed, you can then use `pip` to install `frads`:

	$ python -m pip install frads

### Verifying

To verify that `frads` can be seen by Python, type `python` from your shell. Then at the Python prompt, try to import `frads`

	>>> import frads
	>>> print(frads.__version__)
	1.0.0


## Reference

Wang, T., "Frads: A Python Library for Radiance Simulation Control", 2021 Radiance workshop, Bilbao, Spain, August 19, 2021, [ppt](https://www.radiance-online.org/community/workshops/2021-bilbao-spain-2/presentations/19_thursday/frads.pdf) , [voice recording](https://www.radiance-online.org/community/workshops/2021-bilbao-spain-2)

Wang, T., Ward, G., and Lee, E.S. (2021), A Python Library for Radiance Matrix-based Simulation Control and EnergyPlus Integration, Proceedings of Building Simulation 2021, International Building Performance Simulation Association, Bruges, September 1-3, 2021. Publication to be posted: [pdf](https://www.researchgate.net/publication/358969936_A_Python_Library_for_Radiance_Matrix-based_Simulation_Control_and_EnergyPlus_Integration)
