# gen matrix view-sky
---
```
gen matrix view-sky [-h] [-v] <view_file> <model files...> [options]
```
`gen matrix view-sky` generates the matrix for a two-phase simulation, where the sender
is a grid of views (e.g., a series of workplane sensors) and the receiver is the descritized sky.
By default the sky is in Reinhart sky 4-subdivision, which can be adjusted with the `-b/--basis` option.

## Options
`-b/--basis`
:	Sky basis, can be r1, r2, r4, r6

`-r/--resolu`
:	Image resolution (default: 800)

All `rtrace` options are recognized as well.

## Example

Generate a view to sky matrix (rendering), with a reinhart subdivision 1 sky.
```
gen matrix view-sky view1.vf obj/materials.mat obj/room.rad -b r1
```
