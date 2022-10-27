# gen matrix surface-surface
---
```
gen matrix surface-surface [-h] [-v] <sender_surface> <receiver_surface> \
<model files...> [options]
```
`gen matrix surface-surface` generate a matrix between two arbiturary surface[s]. 
This command can be used, for example, to generate a matrix describes the 
flux transport behavior of a tublar daylighting device.

## Options
`-b/--basis`
:	Sender and receiver sampling basis (default: (kf, kf))

`-f/--offset`
:	Offset the sender and receiver surface in their normal 
directions (default: (0, 0))

All `rtrace` options are recognized as well.

## Example

Generate a surface to surface matrix (e.g., for a tubular daylighting device)
```
gen matrix surface-surface diffuser.rad tdd_dome.rad \
	obj/materials.mat obj/room.rad
```
