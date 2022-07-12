.. rst-class:: hide-header

frads documentation
================

**Version**: |release| 

**Useful links**:
`Source Repository <https://github.com/LBNL-ETA/frads>`__ | `Radiance Homepage <https://www.radiance-online.org>`__ | `Radiance Discourse <https://discourse.radiance-online.org>`__ | `Radiance Tutorial <https://www.radiance-online.org/learning/tutorials>`__

*frads* is an open-source library providing high-level abstraction of Radiance matrix-based simulation workflows.

Matrix algebraic methods reduce the time needed to perform accurate, ray-tracing based, annual daylight simulations by several orders of magnitude. A good deal of expertise is needed however to set up the simulations properly to achieve the desired level of accuracy. *frads* provides users with tools (i.e., *mrad*) that automatically determine which matrix-based method to use then sets the associated simulation parameters, helping beginners learn the different matrix methods by observing the tools' behavior. The user is still required to understand basic concepts underlying matrix-based simulation methods (see `tutorials <https://www.radiance-online.org/learning/tutorials>`_).

Matrix-based methods also enable accurate, ray-tracing generated, irradiance, illuminance, and luminance data to be available for run-time data exchange and co-simulations. *frads* provides users with tools (i.e., *eprad*) that generate the appropriate Radiance-generated data then interfaces with the "actuator" EMS module in EnergyPlus or within the Spawn-of-EnergyPlus and Modelica co-simulation environment.  This enables end users to evaluate the performance of buildings with manual- and automatically-controlled shading and daylighting systems or other site and building features that can change parametrically or on a time-step basis.

Intended audience:

1) Developers who are interested in incorporating multi-phase matrix methods into their software and are seeking examples and guidance; i.e., LBNL-suggested default parameters and settings; and,

2) Engineering firms, researchers, and students who are comfortable working in the command-line or Python scripting environment and tasked with a project that cannot be completed with existing tools.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

User Guide
-------------

This part of the documentation guides you through all of the library's
usage patterns.

.. toctree::
   :maxdepth: 2

   commandline
   library

API Reference
-------------

If you are looking for information on a specific function, class, or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
