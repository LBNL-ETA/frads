# gen matrix surface-sky
---
```
gen matrix surface-sky [-h] [-v] <surface_file> <model files...> [options]
```
`gen matrix surface-sky` generates a matrix that describes the flux-transport behavior 
between the surface and the sky. This matrix is usually the `daylight matrix` in a three-phase simulation. This command takes in a surface object file as the first positional arguments, followed by the rest of the model files.

## Options
`-b/--basis`
:	Sky basis, can be r1, r2, r4, r6

`-f/--offset`
:	Offset the surface in its normal direction (default: 0)

All `rtrace` options are recognized as well.

## Example

Generate a surface to sky matrix (e.g., daylight matrix), with a specific set 
of `rcontrib` options.
```
gen matrix surface-sky window.rad obj/materials.mat obj/room.rad \
	-ab 2 -ad 64 -c 1000
```
