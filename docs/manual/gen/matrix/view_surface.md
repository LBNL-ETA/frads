# gen matrix view-surface
---
```
gen matrix view-surface [-h] [-v] <view_file> <model files...> [options]
```
`gen matrix view-surface` generates a similar view matrix but with a view as a sender.

## Options
`-b/--basis`
:	Surface basis, default: kf

`-r/--resolu`
:	Image resolution (default: 800)

`-f/--offset`
:	Offset surface in its normal direction (default: 0)

All `rtrace` options are recognized as well.

## Example

Generate a view to surface matrix (e.g., image-based view matrix)
```
$ gen matrix view-surface window.rad obj/materials.mat obj/room.rad
```
