# Why use frads

Matrix algebraic methods reduce the time needed to perform accurate,
ray-tracing based, annual daylight simulations by several orders of
magnitude. A good deal of expertise is needed to set up the simulations
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
