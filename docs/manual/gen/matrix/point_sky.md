# gen matrix point-sky
---
```
gen matrix point-sky [-h] [-v] <point_file> <model files...> [options]
```
`gen matrix point-sky` generates the matrix for a two-phase simulation, where the sender
is a grid of points (e.g., a series of workplane sensors) and the receiver is the descritized sky.
By default the sky is in Reinhart sky 4-subdivision, which can be adjusted with the `-b/--basis` option.

## Options
`-b/--basis`
:	Sky basis, can be r1, r2, r4, r6

All `rtrace` options are recognized as well.

## Example

Generate a grid to sky matrix, with a reinher subdivision 4 sky (default), 
and room geometry.
```
gen matrix point-sky grid.pts obj/materials.mat obj/room.rad
```
