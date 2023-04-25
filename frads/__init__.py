"""
Frads is an open-source library providing high-level
abstraction of Radiance matrix-based simulation workflows.

Frads automates setup of these simulations by providing
end users with an open-source, high-level abstraction of
the Radiance command-line workflow (Unix toolbox model),
helping to reduce the steep learning curve and associated
user errors. frads also provides the necessary infrastructure
needed for seamless integration of Radiance and other
modeling tools, such as EnergyPlus.

## Intended audience

1. Developers who are interested in incorporating multi-phase
matrix methods into their software and are seeking examples
and guidance; i.e., LBNL-suggested default parameters and settings; and,
2. Engineering firms, researchers, and students who are comfortable
working in the command-line or Python scripting environment and
tasked with a project that cannot be completed with existing tools.

## Why matrix-based methods?

Matrix algebraic methods reduce the time needed to perform accurate,
ray-tracing based, annual daylight simulations by several orders of
magnitude.

## Why frads?

A good deal of expertise is needed to set up the simulations
properly to achieve the desired level of accuracy.
frads provides users with tools (e.g., `mrad`) that automatically
determine which matrix-based method to use then sets the associated
simulation parameters, helping beginners learn the different matrix
methods by observing the tools’ behavior. The user is still required
to understand basic concepts underlying matrix-based simulation methods
(see [tutorials](https://www.radiance-online.org/learning/tutorials)).

Matrix-based methods also enable accurate, ray-tracing generated,
irradiance, illuminance, and luminance data to be available for
run-time data exchange and co-simulations. frads provides users with
tools that generate the appropriate Radiance-generated
data then interfaces with the “actuator” EMS module in EnergyPlus or
within the Spawn-of-EnergyPlus and Modelica co-simulation environment.
This enables end users to evaluate the performance of buildings with
manual- and automatically-controlled shading and daylighting systems
or other site and building features that can change parametrically
or on a time-step basis.
"""

import logging
import shutil

from .epjson2rad import epjson2rad

from .eprad import load_epmodel, EnergyPlusSetup, ep_datetime_parser

from .matrix import (
    load_matrix,
    multiply_rgb,
    rfluxmtx,
    surface_as_sender,
    points_as_sender,
    view_as_sender,
    surface_as_receiver,
    sun_as_receiver,
    sky_as_receiver,
)

from .parsers import (
    parse_epw,
    parse_wea,
    parse_polygon,
    parse_mrad_config,
)

from .methods import assemble_model, three_phase

from .sky import (
    basis_glow,
    gen_perez_sky,
    genskymtx,
    WeaData,
    WeaMetaData,
)

from .types import (
    View,
)


from .utils import gen_grid, unpack_primitives

from .window import GlazingSystem, AIR, ARGON, KRYPTON, XENON

__version__ = "0.3.1"

logger: logging.Logger = logging.getLogger(__name__)

# Check if Radiance is installed more or less
rad_progs = [
    "rfluxmtx",
    "total",
    "getinfo",
    "pcomb",
    "dctimestep",
    "rmtxop",
    "gendaymtx",
    "rtrace",
]

for prog in rad_progs:
    ppath = shutil.which(prog)
    if ppath is None:
        logger.info("%s not found; check Radiance installation", prog)


__all__ = [
    "AIR",
    "ARGON",
    "assemble_model",
    "basis_glow",
    "EnergyPlusSetup",
    "ep_datetime_parser",
    "epjson2rad",
    "GlazingSystem",
    "gen_perez_sky",
    "gen_grid",
    "genskymtx",
    "KRYPTON",
    "load_epmodel",
    "load_matrix",
    "multiply_rgb",
    "parse_epw",
    "parse_mrad_config",
    "parse_wea",
    "parse_polygon",
    "points_as_sender",
    "rfluxmtx",
    "sky_as_receiver",
    "surface_as_receiver",
    "sun_as_receiver",
    "surface_as_sender",
    "three_phase",
    "unpack_primitives",
    "View",
    "view_as_sender",
    "WeaData",
    "WeaMetaData",
    "XENON",
]
