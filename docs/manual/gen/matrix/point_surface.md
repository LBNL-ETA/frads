# gen matrix point-surface
---
```
gen matrix point-surface [-h] [-v] <point_file> <surface_file> <model files...> [options]
```
`gen matrix point-surface` generates a matrix that describes the 
flux-transport behavior between a grid of sensor and a surface, 
e.g., the view-matrix in a three-phase simulation.

## Options
`-b/--basis`
:	Surface basis (default: kf) 

`-f/--offset`
:	Offset surface in its normal direction (default: 0)

All `rtrace` options are recognized as well.

## Example

Generate a point to surface matrix (e.g., view matrix)
```
gen matrix point-surface window.rad obj/materials.mat obj/room.rad
```
