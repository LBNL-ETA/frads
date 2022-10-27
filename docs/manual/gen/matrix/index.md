# gen matrix
---
```
gen matrix [-h] [-v] <sub-command> [options]
```
`matrix` command consists of several sub-commands, each describe a specific type
of matrix generation workflow. A matrix usually describes the flux-transport of
a system, which usually has a light emitter and light receiver. Because we are 
dealing with backwards ray-tracing here, in which we are sending rays from the 
point-of-interests towards the light source, we will call the light emitter 
receiver and light receiver sender.

Senders are usually our point-of-interests. A sender is a grid of points if we are
interested in knowing how much light those grid of points receives. A sender is a
view if we are interseted in knowing the rendered image from that view. A sender is
a surface is we are interseted in know how much light that surface receives.

Receiver are usually our light sources, such as sky, sun, or any surface that can
be modeled as a light emitter (e.g., windows).

Once we defined our senders and receivers, we quickly have a handful of scenarios that 
we need to handle. These scenarios are the system flux-tranport properties that came 
up usually in matrix-based simulation methods. As a results, `matrix` command consists
of a series of sub-commands that handles each scenario. These sub-commands are 
usually in the form of a `sender`-`receiver` pair:

## Commands

- [gen matrix point-sky](point_sky.md)

- [gen matrix view-sky](view_sky.md)

- [gen matrix surface-sky](surface_sky.md)

- [gen matrix point-surface](point_surface.md)

- [gen matrix view-surface](view_surface.md)

- [gen matrix surface-surface](surface_surface.md)

- [gen matrix point-sun](point_sun.md)

- [gen matrix view-sun](view_sun.md)

- [gen matrix ncp](ncp.md)
