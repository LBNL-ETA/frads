![Install + Test](https://github.com/LBNL-ETA/frads/actions/workflows/main.yml/badge.svg)
![CodeQL](https://github.com/LBNL-ETA/frads/actions/workflows/codeql-analysis.yml/badge.svg)
[![Upload Python Package](https://github.com/LBNL-ETA/frads/actions/workflows/python-publish.yml/badge.svg)](https://github.com/LBNL-ETA/frads/actions/workflows/python-publish.yml)
# _frads_: Framework for Radiance simulation control

This is the repository for _frads_ development. Radiance is a free and open-source, raytracing-based lighting engine that is used extensively by engineering firms for innovative solar control, lighting, and daylighting design to improve the energy efficiency of buildings. With matrix algebraic methods, climate-based annual simulations can now be conducted in less than two minutes. _frads_ automates setup of these simulations by providing end users with an open-source, high-level abstraction of the Radiance command-line workflow (Unix toolbox model), helping to reduce the steep learning curve and associated user errors. _frads_ also provides the necessary infrastructure needed for seamless integration of Radiance and other modeling tools, such as EnergyPlus.

[Documentation](https://lbnl-eta.github.io/frads/)

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

### Install Radiance

If you haven't already, you will need to install Radiance. `frads` will check your Radiance
installation, so make sure you install it first.

To install Radiance, visit Radiance Github [repo](https://github.com/LBNL-ETA/Radiance/releases)
 to download the latest release for your operating system.

You can verify that Radiance is installed properly by typing in the command-line:

```
$ rtrace -version
RADIANCE 5.4a ...
```

### Install frads

After you have Python installed, you should have `pip` command available in your shell environment as well. You can then use `pip` to install `frads`:

	$ python -m pip install frads

Alternatively, more recent version of `frads` can be installed directly from github as well. Watch for the passing/failed tag on github to check if the current version passed the tests.:

	$ python -m pip install git+https://github.com/LBNL-ETA/frads

### Verifying

To verify that `frads` can be seen by Python, type `python` from your shell. Then at the Python prompt, try to import `frads`

	>>> import frads
	>>> print(frads.__version__)
	0.2.7

### Optional external library

`Frads` uses Python standard library for all of its functionalities. However, it will take advantage of [Numpy](https://numpy.org) if you have it installed. It will greatly accelerate the matrix multiplication process, especially for progressive simulation workflow.

The [gencolorsky](other_cli.md#gencolorsky) command line tool in `frads` also relies on [libRadTran](http://www.libradtran.org/) a radiative transfer library for computing the spectrally-resolved radiation data. You'd need to install it first to use [gencolorsky](other_cli.md#gencolorsky).

_frads_ runs from the terminal prompt (command line) on Windows, Mac, and Linux OS. Radiance must be [installed](https://www.radiance-online.org/download-install/radiance-source-code/latest-release) prior to use of _frads_.  You can then install _frads_ by entering the following command in your terminal/cmd/powershell:

```
pip install frads
```

You can also install _frads_ from this Github repository using this command:

```
pip install git+https://github.com/LBNL-ETA/frads.git
```

## Reference

Wang, T., "Frads: A Python Library for Radiance Simulation Control", 2021 Radiance workshop, Bilbao, Spain, August 19, 2021, [ppt](https://www.radiance-online.org/community/workshops/2021-bilbao-spain-2/presentations/19_thursday/frads.pdf) , [voice recording](https://www.radiance-online.org/community/workshops/2021-bilbao-spain-2)

Wang, T., Ward, G., and Lee, E.S. (2021), A Python Library for Radiance Matrix-based Simulation Control and EnergyPlus Integration, Proceedings of Building Simulation 2021, International Building Performance Simulation Association, Bruges, September 1-3, 2021. Publication to be posted: [pdf](https://www.researchgate.net/publication/358969936_A_Python_Library_for_Radiance_Matrix-based_Simulation_Control_and_EnergyPlus_Integration)
