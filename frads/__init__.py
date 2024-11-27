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

from .ep2rad import epmodel_to_radmodel
from .eplus import EnergyPlusSetup, ep_datetime_parser, load_energyplus_model
from .eplus_model import EnergyPlusModel
from .matrix import (
    Matrix,
    SensorSender,
    SkyReceiver,
    SunMatrix,
    SunReceiver,
    SurfaceReceiver,
    SurfaceSender,
    ViewSender,
    load_binary_matrix,
    load_matrix,
    matrix_multiply_rgb,
    surfaces_view_factor,
)
from .methods import (
    FivePhaseMethod,
    MaterialConfig,
    Model,
    SceneConfig,
    SensorConfig,
    Settings,
    ThreePhaseMethod,
    TwoPhaseMethod,
    ViewConfig,
    WindowConfig,
    WorkflowConfig,
)
from .sky import WeaData, WeaMetaData, gen_perez_sky, genskymtx, parse_epw, parse_wea
from .utils import gen_grid, parse_polygon, unpack_primitives
from .window import (
    AIR,
    ARGON,
    KRYPTON,
    XENON,
    Gap,
    Gas,
    GlazingSystem,
    create_glazing_system,
)

__version__ = "1.2.11"

logger: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "AIR",
    "ARGON",
    "EnergyPlusModel",
    "EnergyPlusSetup",
    "FivePhaseMethod",
    "Gap",
    "Gas",
    "GlazingSystem",
    "KRYPTON",
    "Matrix",
    "SensorSender",
    "SkyReceiver",
    "SunMatrix",
    "SunReceiver",
    "SurfaceReceiver",
    "SurfaceSender",
    "ThreePhaseMethod",
    "TwoPhaseMethod",
    "ViewSender",
    "WeaData",
    "WeaMetaData",
    "WorkflowConfig",
    "XENON",
    "create_glazing_system",
    "ep_datetime_parser",
    "epmodel_to_radmodel",
    "gen_grid",
    "gen_perez_sky",
    "genskymtx",
    "load_binary_matrix",
    "load_energyplus_model",
    "load_matrix",
    "matrix_multiply_rgb",
    "parse_epw",
    "parse_polygon",
    "parse_wea",
    "surfaces_view_factor",
    "unpack_primitives",
    "Settings",
    "Model",
    "SceneConfig",
    "ViewConfig",
    "WindowConfig",
    "SensorConfig",
    "MaterialConfig",
]
